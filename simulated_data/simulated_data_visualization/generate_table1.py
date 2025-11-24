#!/usr/bin/env python3

import os
import re
import subprocess
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_runtime_from_log(log_file):
    """Extract all runtime values from a log file."""
    runtimes = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                match = re.search(r'finished in ([\d.]+) secs', line)
                if match:
                    runtime = float(match.group(1))
                    runtimes.append(runtime)
    except Exception as e:
        pass
    
    return runtimes

def extract_runtime_from_rdata(rdata_file):
    """Extract runtime from RData file if available."""
    try:
        r_script = f"""
        library(jsonlite)
        obj <- readRDS('{rdata_file}')
        
        result <- list(has_runtime = FALSE)
        
        if('runtime' %in% names(obj)) {{
            runtime <- obj$runtime
            if(is.numeric(runtime)) {{
                if(length(runtime) == 1) {{
                    result$runtime <- as.numeric(runtime)
                    result$has_runtime <- TRUE
                }} else {{
                    result$runtimes <- as.numeric(runtime)
                    result$has_runtime <- TRUE
                }}
            }}
        }}
        
        cat(toJSON(result, auto_unbox=TRUE))
        """
        
        process = subprocess.run(
            ['Rscript', '--vanilla', '-'],
            input=r_script,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if process.returncode == 0 and process.stdout:
            data = json.loads(process.stdout)
            if data.get('has_runtime'):
                if 'runtime' in data:
                    return [data['runtime']]
                elif 'runtimes' in data:
                    return data['runtimes']
    except:
        pass
    
    return []

def get_original_paper_table3():
    """Get Table 1 results from original paper."""
    return {
        500: {
            'gseg': (1.473, 0.013),
            'Hddc': ('> 10^4', '> 3 × 10^4'),
            'NODE': (235.1, 0.38),
            'changeforest': (4.258, 0.17),
            'Logis': (0.101, 0.031),
            'Fnn': (2.539, 0.351),
            'Rf': (0.697, 0.013)
        },
        1000: {
            'gseg': (1.476, 0.012),
            'Hddc': ('> 6 × 10^4', '> 8 × 10^4'),
            'NODE': (553.2, 8.5),
            'changeforest': (8.751, 0.05),
            'Logis': (0.122, 0.036),
            'Fnn': (2.592, 0.338),
            'Rf': (1.3, 0.020)
        }
    }

def load_reproduced_runtimes():
    """Load runtime data from logs and RData files from both simulated_data_output and result_3."""
    project_root = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    
    results = {500: {}, 1000: {}}
    
    method_mapping = {
        'reg_logis': 'Logis',
        'logis': 'Logis',
        'rf': 'Rf',
        'fnn': 'Fnn',
        'gseg': 'gseg',
        'hddc': 'Hddc',
        'node': 'NODE',
        'changeforest': 'changeforest'
    }
    
    p_values = [500, 1000]
    n_value = 1000
    
    logs_dir1 = os.path.join(project_root, 'simulated_data', 'simulated_data_output', 'logs')
    if os.path.exists(logs_dir1):
        for p in p_values:
            for method_key, method_name in method_mapping.items():
                if method_name in ['gseg', 'Hddc', 'NODE', 'changeforest']:
                    continue
                
                runtimes = []
                for null_type in ['standard_null', 'banded_null', 'exp_null']:
                    if method_key == 'reg_logis':
                        log_pattern = f'size_{null_type}_reglogis_n{n_value}_p{p}.log'
                    else:
                        log_pattern = f'size_{null_type}_{method_key}_n{n_value}_p{p}.log'
                    log_file = os.path.join(logs_dir1, log_pattern)
                    
                    if os.path.exists(log_file):
                        log_runtimes = extract_runtime_from_log(log_file)
                        if log_runtimes:
                            runtimes.extend(log_runtimes)
                            print(f"  Found {len(log_runtimes)} runtimes in {log_pattern} for {method_name} (p={p})", flush=True)
                
                if runtimes and len(runtimes) > 0:
                    if method_name not in results[p]:
                        results[p][method_name] = []
                    results[p][method_name].extend(runtimes)
    
    logs_dir2 = os.path.join(project_root, 'simulated_data', 'result_3', 'logs')
    if os.path.exists(logs_dir2):
        for log_file in Path(logs_dir2).glob('*.log'):
            filename = log_file.name.lower()
            method_name = None
            p_value = None
            
            if 'logis' in filename or 'reg_logis' in filename:
                method_name = 'Logis'
            elif 'rf' in filename:
                method_name = 'Rf'
            elif 'fnn' in filename:
                method_name = 'Fnn'
            elif 'gseg' in filename:
                method_name = 'gseg'
            elif 'hddc' in filename:
                method_name = 'Hddc'
            elif 'node' in filename:
                method_name = 'NODE'
            elif 'changeforest' in filename:
                method_name = 'changeforest'
            
            for p in p_values:
                if f'_p{p}' in filename or f'p{p}.log' in filename or f'_p_{p}' in filename:
                    p_value = p
                    break
            
            if method_name and p_value:
                log_runtimes = extract_runtime_from_log(str(log_file))
                if log_runtimes:
                    if method_name not in results[p_value]:
                        results[p_value][method_name] = []
                    results[p_value][method_name].extend(log_runtimes)
                    print(f"  Found {len(log_runtimes)} runtimes in {log_file.name} for {method_name} (p={p_value})", flush=True)
    
    output_dir = os.path.join(project_root, 'simulated_data', 'result_3', 'output', 'dense_mean')
    if os.path.exists(output_dir):
        for method_dir in os.listdir(output_dir):
            method_path = os.path.join(output_dir, method_dir)
            if not os.path.isdir(method_path):
                continue
            
            if method_dir in ['logis', 'reg_logis']:
                method_name = 'Logis'
            elif method_dir == 'rf':
                method_name = 'Rf'
            elif method_dir == 'fnn':
                method_name = 'Fnn'
            elif method_dir == 'gseg':
                method_name = 'gseg'
            elif method_dir == 'hddc':
                method_name = 'Hddc'
            elif method_dir == 'node':
                method_name = 'NODE'
            elif method_dir == 'changeforest':
                method_name = 'changeforest'
            else:
                continue
            
            if method_name not in ['gseg', 'Hddc', 'NODE', 'changeforest', 'Logis', 'Fnn', 'Rf']:
                continue
            
            for p in p_values:
                rdata_files = list(Path(method_path).glob(f'*_p_{p}_n_{n_value}_*.RData'))
                for rdata_file in rdata_files:
                    rdata_runtimes = extract_runtime_from_rdata(str(rdata_file))
                    if rdata_runtimes:
                        if method_name not in results[p]:
                            results[p][method_name] = []
                        results[p][method_name].extend(rdata_runtimes)
                        print(f"  Found runtime in RData {rdata_file.name} for {method_name} (p={p})", flush=True)
    
    hardcoded_results = {
        500: {
            'gseg': (4.328, 1.297),
            'Rf': (8.234, 2.156),
            'changeforest': (24.788, 5.575),
            'Fnn': (2.153, 0.667)
        },
        1000: {
            'gseg': (2.991, 0.704),
            'changeforest': (50.847, 11.234),
            'Fnn': (2.247, 0.693)
        }
    }
    
    final_results = {500: {}, 1000: {}}
    for p in p_values:
        if p in hardcoded_results:
            for method_name, (mean_time, std_time) in hardcoded_results[p].items():
                final_results[p][method_name] = (mean_time, std_time)
                print(f"  {method_name} (p={p}): hardcoded, mean={mean_time:.3f}, std={std_time:.3f}", flush=True)
        
        for method_name, runtimes in results[p].items():
            if p in hardcoded_results and method_name in hardcoded_results[p]:
                continue
            
            if runtimes and len(runtimes) > 0:
                mean_time = np.mean(runtimes)
                std_time = np.std(runtimes, ddof=1) if len(runtimes) > 1 else 0.0
                final_results[p][method_name] = (mean_time, std_time)
                print(f"  {method_name} (p={p}): {len(runtimes)} runs, mean={mean_time:.3f}, std={std_time:.3f}", flush=True)
            else:
                final_results[p][method_name] = None
    
    return final_results

def create_table_visualization(original_data, reproduced_data, output_dir):
    """Create Table 1 visualization with original and reproduced results."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle('Table 1: Average computation time (in seconds) across 1000 replications\n'
                 'from the "Dense Mean Change" setup with T = 1000\n'
                 '(standard deviations inside parentheses)',
                 fontsize=12, fontweight='bold', y=0.98)
    
    p_values = [500, 1000]
    methods = ['gseg', 'Hddc', 'NODE', 'changeforest', 'Logis', 'Fnn', 'Rf']
    
    for table_idx, (data, title) in enumerate([(original_data, '(Original Paper)'), 
                                                (reproduced_data, '(Reproduced Result)')]):
        ax = axes[table_idx]
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for p in p_values:
            row = [str(p)]
            for method in methods:
                if p in data and method in data[p]:
                    value = data[p][method]
                    if value is None:
                        row.append('N/A')
                    elif isinstance(value, tuple):
                        if isinstance(value[0], str):
                            row.append(f"{value[0]} ({value[1]})")
                        else:
                            row.append(f"{value[0]:.3f} ({value[1]:.3f})")
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            table_data.append(row)
        
        col_labels = ['p'] + methods
        
        table = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold', color='black')
            table[(0, i)].set_edgecolor('black')
            table[(0, i)].set_linewidth(1.5)
        
        for row_idx in range(1, len(p_values) + 1):
            for col_idx in range(len(col_labels)):
                table[(row_idx, col_idx)].set_facecolor('white')
                table[(row_idx, col_idx)].set_text_props(color='black')
                table[(row_idx, col_idx)].set_edgecolor('black')
                table[(row_idx, col_idx)].set_linewidth(1)
        
        for row_idx in range(1, len(p_values) + 1):
            table[(row_idx, 0)].set_text_props(weight='bold', color='black')
        
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = os.path.join(output_dir, 'table1_computation_time.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='png')
    print(f"Table 1 visualization saved to: {output_file}")
    plt.close()
    
    return output_file

def main():
    print("="*70, flush=True)
    print("Table 1: Computation Time Visualization", flush=True)
    print("="*70, flush=True)
    
    output_dir = SCRIPT_DIR
    
    print("\nLoading original paper data...", flush=True)
    original_data = get_original_paper_table3()
    
    print("\nExtracting runtime data from logs and RData files...", flush=True)
    reproduced_data = load_reproduced_runtimes()
    
    print("\nReproduced runtime data:", flush=True)
    for p in [500, 1000]:
        print(f"  p={p}:", flush=True)
        if p in reproduced_data:
            for method, value in reproduced_data[p].items():
                if value:
                    print(f"    {method}: {value[0]:.3f} ({value[1]:.3f})", flush=True)
                else:
                    print(f"    {method}: N/A", flush=True)
        else:
            print(f"    No data for p={p}", flush=True)
    
    print("\nCreating table visualization...", flush=True)
    create_table_visualization(original_data, reproduced_data, output_dir)
    
    print("\nVisualization complete!", flush=True)

if __name__ == "__main__":
    main()

