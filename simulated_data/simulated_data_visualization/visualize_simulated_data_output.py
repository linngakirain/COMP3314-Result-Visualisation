#!/usr/bin/env python3
"""Visualize ARI and AUC data from simulated_data_output folder."""

import os
import sys
import json
import pickle
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

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

def load_all_data():
    """Load ARI and AUC data from simulated_data_output folder."""
    output_dir = os.path.join(PROJECT_ROOT, 'simulated_data', 'simulated_data_output', 'output')
    
    results = {}
    null_types = ['standard_null', 'banded_null', 'exp_null']
    methods = ['reg_logis', 'rf', 'fnn']
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return results
    
    for null_type in null_types:
        null_dir = os.path.join(output_dir, null_type)
        if not os.path.exists(null_dir):
            continue
        
        for method in methods:
            method_dir = os.path.join(null_dir, method)
            if not os.path.exists(method_dir):
                continue
            
            for data_file in Path(method_dir).glob('*.RData'):
                filename = data_file.name
                
                try:
                    parts = filename.split('_')
                    if 'p' in parts and 'n' in parts:
                        p_idx = parts.index('p') + 1
                        n_idx = parts.index('n') + 1
                        p = int(parts[p_idx])
                        n = int(parts[n_idx])
                        
                        key = (null_type, method, n, p)
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
    
    return results

def create_ari_visualization(data, output_dir):
    """Create ARI boxplot visualization organized by null type and method."""
    method_labels = {
        'reg_logis': 'Logis',
        'rf': 'RF',
        'fnn': 'FNN'
    }
    
    null_labels = {
        'standard_null': 'Standard Null',
        'banded_null': 'Banded Null',
        'exp_null': 'Exponential Null'
    }
    
    null_types = ['standard_null', 'banded_null', 'exp_null']
    methods = ['reg_logis', 'rf', 'fnn']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('ARI (Adjusted Rand Index) Performance by Null Type', 
                 fontsize=14, fontweight='bold')
    
    method_colors = {
        'Logis': 'blue',
        'RF': 'purple',
        'FNN': 'orange'
    }
    
    for null_idx, null_type in enumerate(null_types):
        ax = axes[null_idx]
        
        plot_data = []
        labels = []
        
        for method in methods:
            method_label = method_labels.get(method, method)
            p_values = sorted(set([key[3] for key in data.keys() 
                                 if isinstance(key, tuple) and len(key) == 4 
                                 and key[0] == null_type and key[1] == method]))
            
            for p in p_values:
                n_values = sorted(set([key[2] for key in data.keys() 
                                     if isinstance(key, tuple) and len(key) == 4 
                                     and key[0] == null_type and key[1] == method and key[3] == p]))
                
                for n in n_values:
                    key = (null_type, method, n, p)
                    if key in data and data[key].get('ari') and len(data[key]['ari']) > 0:
                        plot_data.append(data[key]['ari'])
                        labels.append(f'{method_label}\np={p}, n={n}')
        
        if plot_data:
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, 
                           showmeans=True, meanline=True)
            for i, patch in enumerate(bp['boxes']):
                method_label = labels[i].split('\n')[0]
                color = method_colors.get(method_label, 'gray')
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_ylabel('ARI', fontsize=11, fontweight='bold')
        ax.set_title(null_labels.get(null_type, null_type), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.1])
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        ax.tick_params(axis='y', labelsize=9)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'simulated_output_ari_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"ARI visualization saved to: {output_file}")
    plt.close()

def create_auc_visualization(data, output_dir):
    """Create AUC boxplot visualization organized by null type and method."""
    method_labels = {
        'reg_logis': 'Logis',
        'rf': 'RF',
        'fnn': 'FNN'
    }
    
    null_labels = {
        'standard_null': 'Standard Null',
        'banded_null': 'Banded Null',
        'exp_null': 'Exponential Null'
    }
    
    null_types = ['standard_null', 'banded_null', 'exp_null']
    methods = ['reg_logis', 'rf', 'fnn']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('AUC (Area Under Curve) Performance by Null Type', 
                 fontsize=14, fontweight='bold')
    
    method_colors = {
        'Logis': 'blue',
        'RF': 'purple',
        'FNN': 'orange'
    }
    
    for null_idx, null_type in enumerate(null_types):
        ax = axes[null_idx]
        
        plot_data = []
        labels = []
        
        for method in methods:
            method_label = method_labels.get(method, method)
            p_values = sorted(set([key[3] for key in data.keys() 
                                 if isinstance(key, tuple) and len(key) == 4 
                                 and key[0] == null_type and key[1] == method]))
            
            for p in p_values:
                n_values = sorted(set([key[2] for key in data.keys() 
                                     if isinstance(key, tuple) and len(key) == 4 
                                     and key[0] == null_type and key[1] == method and key[3] == p]))
                
                for n in n_values:
                    key = (null_type, method, n, p)
                    if key in data and data[key].get('max_aucs') and len(data[key]['max_aucs']) > 0:
                        plot_data.append(data[key]['max_aucs'])
                        labels.append(f'{method_label}\np={p}, n={n}')
        
        if plot_data:
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True, 
                           showmeans=True, meanline=True)
            for i, patch in enumerate(bp['boxes']):
                method_label = labels[i].split('\n')[0]
                color = method_colors.get(method_label, 'gray')
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        else:
            ax.text(0.5, 0.5, 'No data available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_ylabel('AUC', fontsize=11, fontweight='bold')
        ax.set_title(null_labels.get(null_type, null_type), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        ax.tick_params(axis='y', labelsize=9)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'simulated_output_auc_boxplot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"AUC visualization saved to: {output_file}")
    plt.close()

def main():
    print("="*70, flush=True)
    print("Simulated Data Output Visualization", flush=True)
    print("="*70, flush=True)
    
    print("\nLoading ARI and AUC data from simulated_data_output...", flush=True)
    data = load_all_data()
    
    print(f"\nFound data for {len(data)} configurations", flush=True)
    for key, values in list(data.items())[:10]:
        if isinstance(key, tuple) and len(key) == 4:
            null_type, method, n, p = key
            ari_count = len(values.get('ari', []))
            auc_count = len(values.get('max_aucs', []))
            print(f"  {null_type}/{method} (n={n}, p={p}): ARI={ari_count}, AUC={auc_count}", flush=True)
    
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

