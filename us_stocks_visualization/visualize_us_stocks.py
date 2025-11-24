#!/usr/bin/env python3
"""
Visualization script for US stocks change-points data.
Extracts data from pkl and RData files and creates visualizations.
"""

import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import os
import subprocess
import json
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def load_pkl_change_points(filepath, method_name, date_index=None):
    """Load change points from a pkl file."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        change_points = []
        
        if isinstance(data, pd.DatetimeIndex):
            change_points = [d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d for d in data]
        elif isinstance(data, tuple) and len(data) >= 2:
            results = data[1] if len(data) > 1 else []
            if date_index is None:
                date_index = load_date_index()
            
            for result in results:
                if isinstance(result, dict):
                    if 'cp' in result:
                        cp_index = int(result['cp'])
                        if 0 <= cp_index < len(date_index):
                            cp_date = date_index[cp_index]
                            change_points.append(cp_date.to_pydatetime() if hasattr(cp_date, 'to_pydatetime') else datetime.fromtimestamp(cp_date.timestamp()))
                    if 'output' in result and isinstance(result['output'], dict):
                        output = result['output']
                        if 'cp' in output:
                            cp_index = int(output['cp'])
                            if 0 <= cp_index < len(date_index):
                                cp_date = date_index[cp_index]
                                change_points.append(cp_date.to_pydatetime() if hasattr(cp_date, 'to_pydatetime') else datetime.fromtimestamp(cp_date.timestamp()))
        
        return sorted(change_points)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []

def load_date_index():
    """Load the date index from stable_stocks.csv to convert indices to dates."""
    possible_paths = [
        os.path.join(PROJECT_ROOT, "real_data/us_stocks/us_stocks/stable_stocks.csv"),
        os.path.join(PROJECT_ROOT, "real_data/us_stocks/stable_stocks.csv"),
        os.path.join(PROJECT_ROOT, "data/us_stocks/stable_stocks.csv"),
    ]
    
    for filepath in possible_paths:
        if os.path.exists(filepath):
            try:
                dat = pd.read_csv(filepath, dtype=float, index_col=0, parse_dates=True)
                dates = dat.index
                print(f"Loaded date index from {filepath}: {len(dates)} dates")
                print(f"  Date range: {dates[0]} to {dates[-1]}")
                return dates
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
    
    print("WARNING: Could not load date index from CSV. Using generated dates from 2005-01-01")
    start_date = datetime(2005, 1, 1)
    dates = pd.date_range(start=start_date, periods=1510, freq='D')
    return dates

def load_rdata_change_points(filepath, method_name, date_index=None):
    """Load change points from an RData file using Rscript.
    
    According to the guide, RData files contain:
    - output: list of results
    - Each item has 'cp': integer (absolute change point index)
    """
    try:
        r_script = f"""
        library(jsonlite)
        
        obj <- readRDS('{filepath}')
        cp_indices <- integer(0)
        
        if(is.list(obj) && 'output' %in% names(obj)) {{
            output <- obj$output
            if(is.list(output) && length(output) > 0) {{
                for(i in 1:length(output)) {{
                    item <- output[[i]]
                    if(is.list(item) && 'cp' %in% names(item)) {{
                        cp <- item$cp
                        if(is.numeric(cp)) {{
                            cp_val <- ifelse(is.null(names(cp)), as.integer(cp[1]), as.integer(cp[1]))
                            cp_indices <- c(cp_indices, cp_val)
                        }}
                    }}
                }}
            }}
        }}
        
        result <- list(change_point_indices = as.integer(unique(cp_indices)))
        cat(toJSON(result, auto_unbox=TRUE))
        """
        
        process = subprocess.run(
            ['Rscript', '--vanilla', '-'],
            input=r_script,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if process.returncode != 0:
            if process.stderr:
                print(f"Rscript error for {filepath}: {process.stderr[:200]}")
            return []
        
        if not process.stdout.strip():
            return []
        
        data = json.loads(process.stdout)
        cp_indices = data.get('change_point_indices', [])
        
        if date_index is None:
            date_index = load_date_index()
        
        change_points = []
        for idx in cp_indices:
            try:
                if 0 <= idx < len(date_index):
                    cp_date = date_index[idx]
                    change_points.append(cp_date.to_pydatetime() if hasattr(cp_date, 'to_pydatetime') else datetime.fromtimestamp(cp_date.timestamp()))
            except Exception as e:
                continue
        
        return sorted(change_points)
    except subprocess.TimeoutExpired:
        print(f"Timeout loading {filepath}")
        return []
    except Exception as e:
        print(f"Error loading RData {filepath}: {e}")
        return []

def load_all_change_points():
    """Load change points from all methods."""
    output_dir = os.path.join(PROJECT_ROOT, "real_data/us_stocks/output")
    
    date_index = load_date_index()
    all_cps = {}
    
    pkl_files = {
        'changeforest': 'changeforest_sbs_us_stocks.pkl',
        'fnn': 'fnn_sbs_us_stocks.pkl'
    }
    
    for method, filename in pkl_files.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            cps = load_pkl_change_points(filepath, method, date_index)
            all_cps[method] = cps
            print(f"{method}: {len(cps)} change points")
    
    rdata_files = {
        'gseg_wei': 'gseg_wei_sbs_us_stocks.RData',
        'gseg_max': 'gseg_maxt_sbs_us_stocks.RData',
        'gseg_orig': 'gseg_orig_sbs_us_stocks.RData',
        'gseg_gen': 'gseg_gen_sbs_us_stocks.RData'
    }
    
    for method, filename in rdata_files.items():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading {method} from {filename}...", flush=True)
            cps = load_rdata_change_points(filepath, method, date_index)
            all_cps[method] = cps
            print(f"{method}: {len(cps)} change points", flush=True)
            if cps:
                for cp in cps[:3]:
                    print(f"    {cp.strftime('%Y-%m-%d')}", flush=True)
    
    return all_cps

def get_original_paper_table5():
    """Get Table 3 results from original paper."""
    return {
        'gseg_wei': {
            2006: '11-29',
            2007: '02-12',
            2008: '05-30',
            2009: '12-02'
        },
        'gseg_max': {
            2006: '12-18',
            2007: '02-12',
            2008: '05-30',
            2009: '11-19'
        },
        'Hddc': {
            2006: '08-09',
            2007: '07-25',
            2008: None,
            2009: '07-13'
        },
        'NODE': {
            2006: None,
            2007: None,
            2008: None,
            2009: None
        },
        'changeforest': {
            2006: '08-10',
            2007: None,
            2008: '01-02',
            2009: '06-04'
        },
        'Rf': {
            2006: '08-09',
            2007: '07-25',
            2008: '09-09',
            2009: '08-07'
        }
    }

def format_change_points_by_year(change_points):
    """Format change points by year in MM-DD format."""
    by_year = {}
    for cp in change_points:
        if isinstance(cp, datetime):
            year = cp.year
            month_day = cp.strftime('%m-%d')
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(month_day)
    
    for year in by_year:
        by_year[year] = sorted(set(by_year[year]))
    
    return by_year

def load_us_stocks_data():
    """Load US stocks time series data from stable_stocks files."""
    possible_paths = [
        os.path.join(PROJECT_ROOT, "real_data/us_stocks/us_stocks/stable_stocks.csv"),
        os.path.join(PROJECT_ROOT, "real_data/us_stocks/stable_stocks.csv"),
        os.path.join(PROJECT_ROOT, "real_data/us_stocks/us_stocks/stable_stocks_no_dates.csv"),
        os.path.join(PROJECT_ROOT, "real_data/us_stocks/stable_stocks_no_dates.csv"),
        os.path.join(PROJECT_ROOT, "us_stocks/stable_stocks_no_dates.csv"),
    ]
    
    for filepath in possible_paths:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    first_line = f.readline()
                    if 'git-lfs' in first_line.lower() or 'oid sha256' in first_line.lower():
                        print(f"WARNING: {filepath} is a Git LFS pointer file, not actual data.")
                        print("   Attempting to use git lfs pull or checking for actual data file...")
                        try:
                            subprocess.run(['git', 'lfs', 'pull', filepath], 
                                         check=False, capture_output=True, timeout=5)
                            with open(filepath, 'r') as f2:
                                first_line2 = f2.readline()
                                if 'git-lfs' in first_line2.lower():
                                    print("   Git LFS pull did not retrieve actual data. Using synthetic data.")
                                    continue
                        except:
                            print("   Git LFS not available. Using synthetic data.")
                            continue
                
                print(f"Loading stock data from: {filepath}")
                data = pd.read_csv(filepath, header=0)
                
                first_col = data.iloc[:, 0]
                if isinstance(first_col.iloc[0], str) and any(c in str(first_col.iloc[0]) for c in ['-', '/']):
                    dates = pd.to_datetime(first_col, errors='coerce')
                    dates = dates[~dates.isna()]
                    
                    stock_cols = data.iloc[:, 1:]
                    stock_data = stock_cols.apply(pd.to_numeric, errors='coerce')
                    values = stock_data.mean(axis=1).values
                    
                    valid_idx = ~dates.isna()
                    dates = dates[valid_idx]
                    values = values[valid_idx]
                    values = values[~np.isnan(values)]
                    
                    if len(dates) == len(values) and len(values) > 0:
                        print(f"Loaded {len(values)} data points")
                        print(f"   Date range: {dates.iloc[0].strftime('%Y-%m-%d')} to {dates.iloc[-1].strftime('%Y-%m-%d')}")
                        print(f"   Value range: {float(values.min()):.4f} to {float(values.max()):.4f}")
                        print(f"   Number of stocks: {len(stock_cols.columns)}")
                        return pd.DataFrame({'date': dates, 'value': values})
                else:
                    if len(data.columns) > 0:
                        values = data.iloc[:, 0].values
                    else:
                        values = data.values.flatten()
                    
                    values = pd.to_numeric(values, errors='coerce')
                    values = values[~np.isnan(values)]
                    
                    if len(values) == 0:
                        print("   No valid numeric data found in file.")
                        continue
                    
                    start_date = datetime(2005, 1, 1)
                    dates = pd.date_range(start=start_date, periods=len(values), freq='D')
                    
                    print(f"Loaded {len(values)} data points")
                    print(f"   Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
                    print(f"   Value range: {float(values.min()):.2f} to {float(values.max()):.2f}")
                    
                    return pd.DataFrame({'date': dates, 'value': values})
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("WARNING: US stocks data file not found or is a Git LFS pointer.")
    print("   Using synthetic data for visualization based on RData intervals (1510 days from 2005-01-01).")
    return generate_us_stocks_data()

def generate_us_stocks_data():
    """Generate synthetic US stocks data for visualization.
    
    Based on RData intervals showing [1, 1510], the data covers approximately
    1510 days starting from 2005-01-01.
    """
    start_date = datetime(2005, 1, 1)
    dates = pd.date_range(start=start_date, periods=1510, freq='D')
    
    np.random.seed(42)
    base_price = 100
    data = []
    
    for i, date in enumerate(dates):
        trend = i * 0.01
        noise = np.random.normal(0, 2)
        seasonal = 5 * np.sin(2 * np.pi * i / 365.25)
        price = base_price + trend + seasonal + noise
        data.append(max(50, price))
    
    return pd.DataFrame({'date': dates, 'value': data})

def create_time_series_plot(stocks_data, all_change_points, output_dir):
    """Create time series plot with change points from all methods."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    ax.plot(stocks_data['date'], stocks_data['value'], 
            linewidth=1, alpha=0.8, color='steelblue', 
            label='Stock Price', zorder=1)
    
    colors = {
        'gseg_wei': 'red',
        'gseg_max': 'orange',
        'changeforest': 'green',
        'fnn': 'purple',
        'gseg_orig': 'brown',
        'gseg_gen': 'pink',
        'Rf': 'cyan'
    }
    
    all_cp_annotations = []
    for method, cps in all_change_points.items():
        color = colors.get(method, 'gray')
        for cp in cps:
            if isinstance(cp, datetime):
                date_idx = (stocks_data['date'] - cp).abs().idxmin()
                date = stocks_data['date'].loc[date_idx]
                cp_value = stocks_data['value'].loc[date_idx]
                
                all_cp_annotations.append({
                    'method': method,
                    'date': date,
                    'value': cp_value,
                    'color': color
                })
    
    all_cp_annotations.sort(key=lambda x: x['date'])
    
    for cp_info in all_cp_annotations:
        method = cp_info['method']
        date = cp_info['date']
        color = cp_info['color']
        cps = all_change_points[method]
        
        ax.axvline(date, color=color, linestyle='--', linewidth=1.5, 
                  alpha=0.7, zorder=2, label=method if date == cps[0] else '')
    
    date_numeric = [mdates.date2num(cp['date']) for cp in all_cp_annotations]
    min_date_dist = (stocks_data['date'].max() - stocks_data['date'].min()).days / 50
    
    y_offsets = []
    for i, cp_info in enumerate(all_cp_annotations):
        nearby_indices = [j for j in range(len(all_cp_annotations)) 
                         if j != i and abs(date_numeric[j] - date_numeric[i]) < min_date_dist]
        
        if nearby_indices:
            before_count = sum(1 for j in nearby_indices if j < i)
            if before_count % 2 == 0:
                y_offset = 30 + (before_count * 15)
            else:
                y_offset = -30 - (before_count * 15)
        else:
            y_offset = 30
        
        y_offsets.append(y_offset)
    
    for i, cp_info in enumerate(all_cp_annotations):
        method = cp_info['method']
        date = cp_info['date']
        cp_value = cp_info['value']
        color = cp_info['color']
        y_offset = y_offsets[i]
        
        if y_offset > 0:
            x_offset = 10
            va = 'bottom'
        else:
            x_offset = 10
            va = 'top'
        
        ax.annotate(f'{method}\n{date.strftime("%Y-%m-%d")}',
                   xy=(date, cp_value),
                   xytext=(x_offset, y_offset), textcoords='offset points',
                   fontsize=7, bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor=color, alpha=0.3, edgecolor=color, linewidth=1),
                   rotation=0, va=va, ha='left', zorder=3)
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stock Price', fontsize=12, fontweight='bold')
    ax.set_title('Figure 4: US Stocks time series', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_xlim([stocks_data['date'].min(), stocks_data['date'].max()])
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'figure4_us_stocks_timeseries.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', format='png')
    print(f"Time series plot saved to: {output_file}")
    plt.close()
    
    return output_file

def create_table_visualization(reproduced_cps, output_dir):
    """Create Table 3 visualization with two separate tables: Original Paper and Reproduced Result."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Table 3: Comparison of detected change-points in US Stock Market data',
                 fontsize=14, fontweight='bold', y=0.98)
    
    years = [2006, 2007, 2008, 2009]
    
    original_data = get_original_paper_table5()
    methods_original = ['gseg_wei', 'gseg_max', 'Hddc', 'NODE', 'changeforest', 'Rf']
    
    ax_original = axes[0]
    ax_original.axis('tight')
    ax_original.axis('off')
    
    table_data_original = []
    for method in methods_original:
        row = [method]
        for year in years:
            value = original_data.get(method, {}).get(year, None)
            row.append(value if value else '-')
        table_data_original.append(row)
    
    col_labels = ['Years'] + [str(year) for year in years]
    
    table_original = ax_original.table(
        cellText=table_data_original,
        colLabels=col_labels,
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table_original.auto_set_font_size(False)
    table_original.set_fontsize(9)
    table_original.scale(1, 2)
    
    for i in range(len(col_labels)):
        table_original[(0, i)].set_facecolor('#E8E8E8')
        table_original[(0, i)].set_text_props(weight='bold', color='black')
        table_original[(0, i)].set_edgecolor('black')
        table_original[(0, i)].set_linewidth(1.5)
    
    for row_idx in range(1, len(methods_original) + 1):
        for col_idx in range(len(col_labels)):
            table_original[(row_idx, col_idx)].set_facecolor('white')
            table_original[(row_idx, col_idx)].set_text_props(color='black')
            table_original[(row_idx, col_idx)].set_edgecolor('black')
            table_original[(row_idx, col_idx)].set_linewidth(1)
    
    for row_idx in range(1, len(methods_original) + 1):
        table_original[(row_idx, 0)].set_text_props(weight='bold', color='black')
    
    ax_original.set_title('(Original Paper)', fontsize=12, fontweight='bold', pad=10)
    
    ax_reproduced = axes[1]
    ax_reproduced.axis('tight')
    ax_reproduced.axis('off')
    
    original_method_order = ['gseg_wei', 'gseg_max', 'Hddc', 'NODE', 'changeforest', 'Rf']
    
    repro_to_orig = {
        'gseg_wei': 'gseg_wei',
        'gseg_max': 'gseg_max',
        'changeforest': 'changeforest'
    }
    
    table_data_reproduced = []
    for orig_method in original_method_order:
        repro_method = None
        for key, mapped in repro_to_orig.items():
            if mapped == orig_method:
                repro_method = key
                break
        
        if repro_method and repro_method in reproduced_cps:
            row = [orig_method]
            cps_by_year = format_change_points_by_year(reproduced_cps[repro_method])
            
            for year in years:
                repro_value = cps_by_year.get(year, [])
                if repro_value:
                    repro_str = ', '.join(repro_value)
                else:
                    repro_str = '-'
                row.append(repro_str)
            
            table_data_reproduced.append(row)
        else:
            row = [orig_method] + ['-'] * len(years)
            table_data_reproduced.append(row)
    
    additional_methods = []
    for method in reproduced_cps.keys():
        if method not in repro_to_orig:
            additional_methods.append(method)
    
    for method in sorted(additional_methods):
        row = [method]
        cps_by_year = format_change_points_by_year(reproduced_cps[method])
        for year in years:
            repro_value = cps_by_year.get(year, [])
            if repro_value:
                repro_str = ', '.join(repro_value)
            else:
                repro_str = '-'
            row.append(repro_str)
        table_data_reproduced.append(row)
    
    if not table_data_reproduced:
        table_data_reproduced = [['No data', '-', '-', '-', '-']]
    
    table_reproduced = ax_reproduced.table(
        cellText=table_data_reproduced,
        colLabels=col_labels,
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table_reproduced.auto_set_font_size(False)
    table_reproduced.set_fontsize(9)
    table_reproduced.scale(1, 2)
    
    for i in range(len(col_labels)):
        table_reproduced[(0, i)].set_facecolor('#E8E8E8')
        table_reproduced[(0, i)].set_text_props(weight='bold', color='black')
        table_reproduced[(0, i)].set_edgecolor('black')
        table_reproduced[(0, i)].set_linewidth(1.5)
    
    for row_idx in range(1, len(table_data_reproduced) + 1):
        for col_idx in range(len(col_labels)):
            table_reproduced[(row_idx, col_idx)].set_facecolor('white')
            table_reproduced[(row_idx, col_idx)].set_text_props(color='black')
            table_reproduced[(row_idx, col_idx)].set_edgecolor('black')
            table_reproduced[(row_idx, col_idx)].set_linewidth(1)
    
    for row_idx in range(1, len(table_data_reproduced) + 1):
        table_reproduced[(row_idx, 0)].set_text_props(weight='bold', color='black')
    
    ax_reproduced.set_title('(Reproduced Result)', fontsize=12, fontweight='bold', pad=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_file = os.path.join(output_dir, 'table3_us_stocks.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='png')
    print(f"Table 3 visualization saved to: {output_file}", flush=True)
    print(f"  Reproduced table has {len(table_data_reproduced)} rows", flush=True)
    for i, row in enumerate(table_data_reproduced[:3]):
        print(f"    Row {i}: {row[0]} - {row[1:]}", flush=True)
    plt.close()
    
    return output_file

def main():
    print("="*70, flush=True)
    print("US Stocks Change-Points Visualization", flush=True)
    print("="*70, flush=True)
    
    output_dir = SCRIPT_DIR
    
    print("\nLoading change points from all methods...", flush=True)
    all_change_points = load_all_change_points()
    
    print("\nChange points loaded:", flush=True)
    for method, cps in sorted(all_change_points.items()):
        print(f"  {method}: {len(cps)} change points", flush=True)
        if cps:
            for cp in cps[:5]:
                print(f"    {cp}", flush=True)
    
    print("\nLoading US stocks time series data...", flush=True)
    stocks_data = load_us_stocks_data()
    print(f"Loaded {len(stocks_data)} data points", flush=True)
    print(f"   Date range: {stocks_data['date'].min()} to {stocks_data['date'].max()}", flush=True)
    
    print("\nCreating visualizations...", flush=True)
    create_time_series_plot(stocks_data, all_change_points, output_dir)
    create_table_visualization(all_change_points, output_dir)
    
    print("\nVisualization complete!", flush=True)
    print(f"Output files saved to: {output_dir}")

if __name__ == "__main__":
    main()

