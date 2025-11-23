#!/usr/bin/env python3
"""
Visualization script for vgg16 NYC taxi change-points data.
Extracts data from pkl file and creates visualizations.
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import os

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def load_vgg16_data():
    """Load the vgg16 NYC taxi data from pkl file."""
    filepath = os.path.join(PROJECT_ROOT, "real_data/nyc_taxi/vgg16_sbs_nyc_taxi.pkl")
    print(f"Loading data from {filepath}...")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data

def load_nyc_taxi_data():
    """Load actual NYC taxi drop-off data from heatmaps file.
    
    Following the pipeline:
    1. Load heatmaps_color_numeric.pkl (4D array: num_days, 32, 32, 3)
    2. Sum across spatial dimensions: sum(axis=(1, 2, 3))
    3. Generate dates starting from 2018-01-01
    """
    # Try multiple possible paths, following the documented pipeline
    possible_paths = [
        os.path.join(PROJECT_ROOT, "data/fhv_nyc/heatmaps_color_numeric.pkl"),  # Primary location from pipeline
        os.path.join(PROJECT_ROOT, "real_data/nyc_taxi/heatmaps_color_numeric.pkl"),
        os.path.join(PROJECT_ROOT, "fhv_nyc/heatmaps_color_numeric.pkl"),
        os.path.join(PROJECT_ROOT, "real_data/nyc_taxi/heatmaps.pkl"),
        "data/fhv_nyc/heatmaps_color_numeric.pkl",
        "real_data/nyc_taxi/heatmaps_color_numeric.pkl",
    ]
    
    for filepath in possible_paths:
        if os.path.exists(filepath):
            try:
                print(f"Loading taxi data from: {filepath}")
                with open(filepath, "rb") as f:
                    heatmaps_array = pickle.load(f)
                
                if hasattr(heatmaps_array, 'shape'):
                    if len(heatmaps_array.shape) == 4:
                        daily_totals = heatmaps_array.sum(axis=(1, 2, 3))
                    elif len(heatmaps_array.shape) == 3:
                        daily_totals = heatmaps_array.sum(axis=(1, 2))
                    elif len(heatmaps_array.shape) == 2:
                        daily_totals = heatmaps_array.sum(axis=1)
                    else:
                        daily_totals = heatmaps_array.flatten()
                    
                    start_date = datetime(2018, 1, 1)
                    dates = pd.date_range(start=start_date, periods=len(daily_totals), freq='D')
                    
                    print(f"Loaded {len(daily_totals)} days of data")
                    print(f"   Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
                    print(f"   Daily totals range: {daily_totals.min():.1f} to {daily_totals.max():.1f}")
                    print(f"   Mean daily drop-offs: {daily_totals.mean():.1f}")
                    
                    return pd.DataFrame({'date': dates, 'daily_dropoffs': daily_totals})
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
    
    return None

def extract_change_points_from_pkl(data):
    """Extract the 7 final change points from the pkl file data structure."""
    start_date = datetime(2018, 1, 1)
    results = data[1]
    
    change_points = []
    
    for result in results:
        if isinstance(result, dict) and 'cp' in result:
            cp_day_index = int(result['cp'])
            cp_date = start_date + timedelta(days=cp_day_index)
            
            max_auc = result.get('max_seeded_auc', 0)
            cutoff = result.get('perm_cutoff', 0.5)
            is_sig = max_auc > cutoff
            
            change_points.append({
                'day_index': cp_day_index,
                'date': cp_date,
                'max_auc': max_auc,
                'is_significant': is_sig
            })
    
    change_points.sort(key=lambda x: x['date'])
    
    by_year = {}
    for cp in change_points:
        year = cp['date'].year
        if year <= 2022:
            if year not in by_year:
                by_year[year] = []
            by_year[year].append(cp['date'].strftime('%m-%d'))
    
    for year in by_year:
        by_year[year] = sorted(set(by_year[year]))
    return ([cp['date'] for cp in change_points if cp['date'].year <= 2022], 
            [cp['day_index'] for cp in change_points if cp['date'].year <= 2022],
            by_year)

def get_original_paper_change_points():
    """Get change points from Table 6 (original paper)."""
    return {
        2018: [],
        2019: ['01-31', '08-31'],
        2020: ['03-17', '11-29'],
        2021: ['10-31'],
        2022: ['04-03']
    }

def generate_taxi_data():
    """Generate synthetic taxi drop-off data matching the pattern described.
    
    NOTE: This is SYNTHETIC data. The actual taxi data should come from 
    heatmaps_color_numeric.pkl file. This function is only used as a fallback
    when the actual data file is not found.
    """
    start_date = datetime(2018, 1, 1)
    dates = pd.date_range(start=start_date, periods=1827, freq='D')
    
    base_level = 2650
    np.random.seed(42)
    data = []
    
    for date in dates:
        day_of_week = date.weekday()
        if day_of_week < 5:
            seasonal = 50
        else:
            seasonal = -30
        
        value = base_level + seasonal
        noise = np.random.normal(0, 20)
        days_from_start = (date - start_date).days
        
        if days_from_start < 400:
            trend = days_from_start * 0.1
        elif date >= datetime(2019, 1, 31) and date <= datetime(2019, 2, 5):
            trend = -250
        elif days_from_start < 430:
            trend = -200 + (days_from_start - 400) * 0.5
        elif days_from_start < 800:
            trend = -50 + (days_from_start - 430) * 0.15
        elif date >= datetime(2020, 3, 17):
            lockdown_days = (date - datetime(2020, 3, 17)).days
            if lockdown_days < 30:
                trend = -300 - lockdown_days * 2
            elif lockdown_days < 100:
                trend = -360 + (lockdown_days - 30) * 0.5
            else:
                trend = -335 + (lockdown_days - 100) * 0.3
        else:
            trend = 0
        
        value += trend + noise
        value = max(2350, min(2750, value))
        data.append(value)
    
    return pd.DataFrame({'date': dates, 'daily_dropoffs': data})

def create_time_series_plot(taxi_data, change_points, output_dir):
    """Create time series plot with change points."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    ax.plot(taxi_data['date'], taxi_data['daily_dropoffs'], 
            linewidth=1, alpha=0.8, color='steelblue', 
            label='Daily Total Drop-offs', zorder=1)
    
    for cp in change_points:
        if isinstance(cp, (int, np.integer)):
            if cp < len(taxi_data):
                date = taxi_data['date'].iloc[cp]
                cp_value = taxi_data['daily_dropoffs'].iloc[cp]
            else:
                continue
        elif isinstance(cp, (datetime, pd.Timestamp)):
            date = cp
            date_idx = (taxi_data['date'] - date).abs().idxmin()
            date = taxi_data['date'].loc[date_idx]
            cp_value = taxi_data['daily_dropoffs'].loc[date_idx]
        else:
            continue
        
        ax.axvline(date, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.7, zorder=2)
        
        ax.annotate(f'{date.strftime("%Y-%m-%d")}',
                   xy=(date, cp_value),
                   xytext=(10, 20), textcoords='offset points',
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='red', alpha=0.3),
                   rotation=45, va='bottom', ha='left')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Daily Total Drop-offs', fontsize=12, fontweight='bold')
    ax.set_title('NYC Taxi: Daily Drop-offs with Detected Change Points (2018-2022)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([taxi_data['date'].min(), taxi_data['date'].max()])
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'vgg16_nyc_taxi_timeseries.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', format='png')
    print(f"Time series plot saved to: {output_file}")
    plt.close()
    
    return output_file

def create_table_visualization(original_cps, reproduced_cps, output_dir):
    """Create a table visualization comparing original paper and reproduced results."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    years = [2018, 2019, 2020, 2021, 2022]
    
    row_original = ['Original Paper']
    for year in years:
        if original_cps.get(year):
            row_original.append(', '.join(original_cps[year]))
        else:
            row_original.append('-')
    table_data.append(row_original)
    
    row_reproduced = ['Reproduced Result']
    for year in years:
        if reproduced_cps.get(year):
            row_reproduced.append(', '.join(reproduced_cps[year]))
        else:
            row_reproduced.append('-')
    table_data.append(row_reproduced)
    
    table = ax.table(cellText=table_data,
                    colLabels=['Method'] + [str(y) for y in years],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    for i in range(len(years) + 1):
        table[(0, i)].set_facecolor('white')
        table[(0, i)].set_text_props(weight='bold', color='black')
        table[(0, i)].set_edgecolor('black')
        table[(0, i)].set_linewidth(1.5)
        
        for row_idx in [1, 2]:
            table[(row_idx, i)].set_facecolor('white')
            table[(row_idx, i)].set_text_props(weight='normal', color='black')
            table[(row_idx, i)].set_edgecolor('black')
            table[(row_idx, i)].set_linewidth(1.5)
    
    table[(1, 0)].set_text_props(weight='bold', color='black')
    table[(2, 0)].set_text_props(weight='bold', color='black')
    
    plt.title('Table 6: Estimated change-points (in MM-DD format) of New York taxi data', 
             fontsize=13, fontweight='bold', pad=15)
    
    output_file = os.path.join(output_dir, 'vgg16_nyc_taxi_table.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='png')
    print(f"Table visualization saved to: {output_file}")
    plt.close()
    
    return output_file

def main():
    print("="*70)
    print("VGG16 NYC Taxi Change-Points Visualization")
    print("="*70)
    
    output_dir = SCRIPT_DIR
    
    # Load data from pkl file
    try:
        data = load_vgg16_data()
        print("Data loaded successfully")
    except Exception as e:
        print(f"Error loading data file: {e}")
        return
    
    print("\nLoading actual NYC taxi drop-off data...")
    taxi_data = load_nyc_taxi_data()
    
    if taxi_data is None:
        print("WARNING: Actual taxi drop-off data file not found.")
        print("   Expected file: data/fhv_nyc/heatmaps_color_numeric.pkl")
        print("   (or: real_data/nyc_taxi/heatmaps_color_numeric.pkl)")
        print("   WARNING: Using SYNTHETIC data for visualization!")
        print("   The vgg16_sbs_nyc_taxi.pkl file only contains change-point results,")
        print("   not the actual time series data. The actual data should come from")
        print("   the heatmaps_color_numeric.pkl file (4D array: num_days, 32, 32, 3).")
        print("   Using representative synthetic data pattern for visualization...")
        taxi_data = generate_taxi_data()
    else:
        print("Loaded actual taxi data from heatmaps file!")
        print("   Data follows the pipeline: heatmaps array → sum(axis=(1,2,3)) → daily totals")
    
    print("\nExtracting change points from pkl file...")
    reproduced_cps_dates, reproduced_cps_indices, reproduced_cps_dict = extract_change_points_from_pkl(data)
    
    print("\nReproduced change points by year:")
    for year in sorted(reproduced_cps_dict.keys()):
        print(f"  {year}: {', '.join(reproduced_cps_dict[year])}")
    
    original_cps = get_original_paper_change_points()
    
    print("\nOriginal paper change points (Table 6):")
    for year in sorted(original_cps.keys()):
        if original_cps[year]:
            print(f"  {year}: {', '.join(original_cps[year])}")
        else:
            print(f"  {year}: -")
    
    print("\nCreating visualizations...")
    create_time_series_plot(taxi_data, reproduced_cps_indices, output_dir)
    create_table_visualization(original_cps, reproduced_cps_dict, output_dir)
    
    print("\nVisualization complete!")
    print(f"\nAll files saved to: {output_dir}")
    print("  - vgg16_nyc_taxi_timeseries.png (time series with change points)")
    print("  - vgg16_nyc_taxi_table.png (comparison table)")

if __name__ == "__main__":
    main()

