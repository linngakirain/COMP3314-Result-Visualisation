#!/usr/bin/env python3

import os
import subprocess
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TRUE_CP = 500

def extract_ari_auc_from_rdata(rdata_file):
    """Extract ARI and AUC data from RData file."""
    try:
        r_script = f"""
        library(jsonlite)
        obj <- readRDS('{rdata_file}')
        result <- list()
        
        if(is.list(obj)) {{
            if('ari' %in% names(obj)) {{
                ari <- obj$ari
                if(is.numeric(ari)) {{
                    result$ari <- as.numeric(ari)
                }} else if(is.list(ari) || is.vector(ari)) {{
                    result$ari <- as.numeric(unlist(ari))
                }}
            }}
            
            if('max_aucs' %in% names(obj)) {{
                max_aucs <- obj$max_aucs
                if(is.numeric(max_aucs)) {{
                    result$max_aucs <- as.numeric(max_aucs)
                }} else if(is.list(max_aucs) || is.vector(max_aucs)) {{
                    result$max_aucs <- as.numeric(unlist(max_aucs))
                }}
            }}
            
            if('output' %in% names(obj) && is.list(obj$output)) {{
                if('ari' %in% names(obj$output)) {{
                    ari <- obj$output$ari
                    if(is.numeric(ari)) {{
                        result$ari <- as.numeric(ari)
                    }} else if(is.list(ari) || is.vector(ari)) {{
                        result$ari <- as.numeric(unlist(ari))
                    }}
                }}
                
                if('max_aucs' %in% names(obj$output)) {{
                    max_aucs <- obj$output$max_aucs
                    if(is.numeric(max_aucs)) {{
                        result$max_aucs <- as.numeric(max_aucs)
                    }} else if(is.list(max_aucs) || is.vector(max_aucs)) {{
                        result$max_aucs <- as.numeric(unlist(max_aucs))
                    }}
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
            timeout=10
        )
        
        if process.returncode == 0 and process.stdout:
            data = json.loads(process.stdout)
            return data
    except Exception as e:
        pass
    
    return None

def extract_ari_auc_from_pkl(pkl_file):
    """Extract ARI and AUC data from pkl file."""
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            
        result = {}
        if isinstance(data, dict):
            if 'ari' in data:
                ari = data['ari']
                if isinstance(ari, (list, np.ndarray)):
                    result['ari'] = np.array(ari).flatten().tolist()
                elif isinstance(ari, (int, float)):
                    result['ari'] = [float(ari)]
            
            if 'max_aucs' in data:
                max_aucs = data['max_aucs']
                if isinstance(max_aucs, (list, np.ndarray)):
                    result['max_aucs'] = np.array(max_aucs).flatten().tolist()
                elif isinstance(max_aucs, (int, float)):
                    result['max_aucs'] = [float(max_aucs)]
        
        return result if result else None
    except Exception as e:
        pass
    
    return None

def load_all_ari_auc_data():
    """Load ARI and AUC data from dense_mean folder only, filter for p=500."""
    dense_mean_dir = os.path.join(PROJECT_ROOT, 'simulated_data', 'computational_time', 'output', 'dense_mean')
    
    target_methods = ['changeforest', 'fnn', 'gseg', 'logis', 'reg_logis', 'rf']
    results = {}
    
    if os.path.exists(dense_mean_dir):
        for method_dir in Path(dense_mean_dir).iterdir():
            if not method_dir.is_dir():
                continue
            
            method_name = method_dir.name
            if method_name not in target_methods:
                continue
            
            for data_file in method_dir.glob('*.RData'):
                filename = data_file.name
                
                try:
                    parts = filename.split('_')
                    if 'p' in parts and 'n' in parts:
                        p_idx = parts.index('p') + 1
                        n_idx = parts.index('n') + 1
                        p = int(parts[p_idx])
                        n = int(parts[n_idx])
                        
                        if p != 500:
                            continue
                        
                        key = (method_name, n, p)
                        if key not in results:
                            results[key] = {'ari': [], 'max_aucs': []}
                        
                        data = extract_ari_auc_from_rdata(str(data_file))
                        if data:
                            if 'ari' in data:
                                ari_data = data['ari']
                                if isinstance(ari_data, list):
                                    results[key]['ari'].extend(ari_data)
                                else:
                                    results[key]['ari'].append(ari_data)
                            if 'max_aucs' in data:
                                auc_data = data['max_aucs']
                                if isinstance(auc_data, list):
                                    results[key]['max_aucs'].extend(auc_data)
                                else:
                                    results[key]['max_aucs'].append(auc_data)
                except (ValueError, IndexError):
                    continue
            
            for data_file in method_dir.glob('*.pkl'):
                filename = data_file.name
                
                try:
                    parts = filename.split('_')
                    if 'p' in parts and 'n' in parts:
                        p_idx = parts.index('p') + 1
                        n_idx = parts.index('n') + 1
                        p = int(parts[p_idx])
                        n = int(parts[n_idx])
                        
                        if p != 500:
                            continue
                        
                        key = (method_name, n, p)
                        if key not in results:
                            results[key] = {'ari': [], 'max_aucs': []}
                        
                        data = extract_ari_auc_from_pkl(str(data_file))
                        if data:
                            if 'ari' in data:
                                ari_data = data['ari']
                                if isinstance(ari_data, list):
                                    results[key]['ari'].extend(ari_data)
                                else:
                                    results[key]['ari'].append(ari_data)
                            if 'max_aucs' in data:
                                auc_data = data['max_aucs']
                                if isinstance(auc_data, list):
                                    results[key]['max_aucs'].extend(auc_data)
                                else:
                                    results[key]['max_aucs'].append(auc_data)
                except (ValueError, IndexError):
                    continue
    
    return results

def create_ari_visualization(data, output_dir):
    """Create ARI boxplot with changeforest, fnn, gseg, logis, and rf (p=500 only)."""
    method_labels = {
        'reg_logis': 'Logis',
        'logis': 'Logis',
        'rf': 'RF',
        'fnn': 'FNN',
        'gseg': 'gseg',
        'changeforest': 'changeforest'
    }
    
    target_methods = ['changeforest', 'fnn', 'gseg', 'logis', 'reg_logis', 'rf']
    consolidated_data = {}
    
    for key in data.keys():
        if isinstance(key, tuple) and len(key) == 3:
            method, n, p = key
            if method not in target_methods or p != 500:
                continue
            
            display_method = method_labels.get(method, method)
            if method in ['logis', 'reg_logis']:
                display_method = 'Logis'
            
            if display_method not in consolidated_data:
                consolidated_data[display_method] = {'ari': []}
            
            if data[key].get('ari') and len(data[key]['ari']) > 0:
                consolidated_data[display_method]['ari'].extend(data[key]['ari'])
    
    methods_with_data = [m for m in ['changeforest', 'Logis', 'FNN', 'gseg', 'RF'] 
                        if m in consolidated_data and len(consolidated_data[m]['ari']) > 0]
    
    if not methods_with_data:
        print("No ARI data found for target methods")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('ARI (Adjusted Rand Index) Performance (p=500)', 
                 fontsize=14, fontweight='bold')
    
    plot_data = []
    labels = []
    
    method_colors = {
        'changeforest': 'red',
        'FNN': 'orange',
        'gseg': 'green',
        'Logis': 'blue',
        'RF': 'purple'
    }
    
    method_order = ['changeforest', 'Logis', 'FNN', 'gseg', 'RF']
    for method in method_order:
        if method in consolidated_data and len(consolidated_data[method]['ari']) > 0:
            plot_data.append(consolidated_data[method]['ari'])
            labels.append(method)
    
    if plot_data:
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, 
                       showmeans=True, meanline=True)
        for i, patch in enumerate(bp['boxes']):
            method_name = labels[i]
            color = method_colors.get(method_name, 'gray')
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_ylabel('ARI', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=9)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'ari_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"ARI visualization saved to: {output_file}")
    plt.close()

def create_auc_visualization(data, output_dir):
    """Create AUC boxplot with changeforest, fnn, gseg, logis, and rf (p=500 only)."""
    method_labels = {
        'reg_logis': 'Logis',
        'logis': 'Logis',
        'rf': 'RF',
        'fnn': 'FNN',
        'gseg': 'gseg',
        'changeforest': 'changeforest'
    }
    
    target_methods = ['changeforest', 'fnn', 'gseg', 'logis', 'reg_logis', 'rf']
    consolidated_data = {}
    
    for key in data.keys():
        if isinstance(key, tuple) and len(key) == 3:
            method, n, p = key
            if method not in target_methods or p != 500:
                continue
            
            display_method = method_labels.get(method, method)
            if method in ['logis', 'reg_logis']:
                display_method = 'Logis'
            
            if display_method not in consolidated_data:
                consolidated_data[display_method] = {'max_aucs': []}
            
            if data[key].get('max_aucs') and len(data[key]['max_aucs']) > 0:
                consolidated_data[display_method]['max_aucs'].extend(data[key]['max_aucs'])
    
    methods_with_data = [m for m in ['changeforest', 'Logis', 'FNN', 'gseg', 'RF'] 
                        if m in consolidated_data and len(consolidated_data[m]['max_aucs']) > 0]
    
    if not methods_with_data:
        print("No AUC data found for target methods")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('AUC (Area Under Curve) Performance (p=500)', 
                 fontsize=14, fontweight='bold')
    
    plot_data = []
    labels = []
    
    method_colors = {
        'changeforest': 'red',
        'Logis': 'blue',
        'FNN': 'orange',
        'gseg': 'green',
        'RF': 'purple'
    }
    
    method_order = ['changeforest', 'Logis', 'FNN', 'gseg', 'RF']
    for method in method_order:
        if method in consolidated_data and len(consolidated_data[method]['max_aucs']) > 0:
            plot_data.append(consolidated_data[method]['max_aucs'])
            labels.append(method)
    
    if plot_data:
        bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, 
                       showmeans=True, meanline=True)
        for i, patch in enumerate(bp['boxes']):
            method_name = labels[i]
            color = method_colors.get(method_name, 'gray')
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    else:
        ax.text(0.5, 0.5, 'No AUC data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.set_ylabel('AUC', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=9)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'auc_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"AUC visualization saved to: {output_file}")
    plt.close()

def main():
    print("="*70, flush=True)
    print("ARI and AUC Visualization", flush=True)
    print("="*70, flush=True)
    
    print("\nLoading ARI and AUC data from simulated data files...", flush=True)
    data = load_all_ari_auc_data()
    
    print(f"\nFound data for {len(data)} configurations", flush=True)
    for key, values in list(data.items())[:10]:
        if isinstance(key, tuple) and len(key) == 3:
            method, n, p = key
            ari_count = len(values.get('ari', []))
            auc_count = len(values.get('max_aucs', []))
            print(f"  {method} (n={n}, p={p}): ARI={ari_count}, AUC={auc_count}", flush=True)
    
    if len(data) == 0:
        print("WARNING: No data found! Check data files.", flush=True)
        return
    
    print("\nCreating ARI visualization...", flush=True)
    create_ari_visualization(data, SCRIPT_DIR)
    
    print("\nCreating AUC visualization...", flush=True)
    create_auc_visualization(data, SCRIPT_DIR)
    
    print("\nVisualization complete!", flush=True)

if __name__ == "__main__":
    main()

