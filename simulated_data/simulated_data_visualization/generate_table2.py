import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def load_cifar_data():
    cifar_dir = Path(PROJECT_ROOT) / 'simulated_data' / 'cifar' / 'vgg16'
    
    if not cifar_dir.exists():
        print(f"CIFAR directory not found: {cifar_dir}")
        return {}
    
    results = defaultdict(lambda: {
        'max_aucs': [],
        'aris': [],
        'aucs': []
    })
    
    pkl_files = sorted(cifar_dir.glob('*.pkl'))
    print(f"Found {len(pkl_files)} CIFAR VGG16 files")
    
    for pkl_file in pkl_files:
        try:
            filename = pkl_file.name
            parts = filename.replace('.pkl', '').split('_')
            
            case = None
            for i, part in enumerate(parts):
                if '-' in part and part.replace('-', '').isdigit():
                    case = part
                    break
            
            if case is None:
                case = 'unknown'
            
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            if 'max_auc' in data:
                max_auc = float(data['max_auc'])
                results[case]['max_aucs'].append(max_auc)
            
            if 'ari' in data:
                ari = float(data['ari'])
                results[case]['aris'].append(ari)
            
            if 'auc' in data:
                auc_array = np.array(data['auc'])
                if auc_array.size > 0:
                    results[case]['aucs'].extend(auc_array.flatten().tolist())
        
        except Exception as e:
            print(f"Error loading {pkl_file.name}: {e}")
            continue
    
    return dict(results)

def calculate_size_performance(all_max_aucs):
    if not all_max_aucs or len(all_max_aucs) == 0:
            return None
            
    all_max_aucs_array = np.array(all_max_aucs)
    critical_val = np.quantile(all_max_aucs_array, 0.95)
    size = np.mean(all_max_aucs_array > critical_val) * 100
    return round(size, 1)

def calculate_power_performance(all_max_aucs):
    if not all_max_aucs or len(all_max_aucs) == 0:
        return None

    all_max_aucs_array = np.array(all_max_aucs)
    critical_val = np.quantile(all_max_aucs_array, 0.95)
    power = np.mean(all_max_aucs_array > critical_val) * 100
    return round(power, 1)

def get_original_paper_table1():
    return {
        'vgg16': {
            'Size': 4.0,
            'Power': 3.4,
            'ARI_Cat_Dog': 0.965,
            'ARI_Deer_Dog': 0.997,
            'ARI_Deer_Horse': 0.994
        }
    }

def create_table_visualization(reproduced_data, original_data, output_dir):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis('tight')
    ax.axis('off')
    
    case_mapping = {
        '3-5': 'Cat → Dog',
        '4-5': 'Deer → Dog',
        '4-7': 'Deer → Horse'
    }
    
    rows = []
    
    if 'vgg16' in reproduced_data:
        method_data = reproduced_data['vgg16']
        size = method_data.get('Size')
        power = method_data.get('Power')
        ari_values = method_data.get('ARI', {})
        
        rows.append(['Size', f"{size}%" if size is not None else "N/A"])
        rows.append(['Power', f"{power}%" if power is not None else "N/A"])
        
        for case_key, case_label in case_mapping.items():
            if case_key in ari_values:
                rows.append([case_label, f"{ari_values[case_key]:.3f}"])
            else:
                rows.append([case_label, "N/A"])
    
    if original_data and 'vgg16' in original_data:
        orig_data = original_data['vgg16']
        orig_rows = []
        orig_rows.append(['Size', f"{orig_data.get('Size', 'N/A')}%"])
        orig_rows.append(['Power', f"{orig_data.get('Power', 'N/A')}%"])
        orig_rows.append(['Cat → Dog', f"{orig_data.get('ARI_Cat_Dog', 'N/A'):.3f}"])
        orig_rows.append(['Deer → Dog', f"{orig_data.get('ARI_Deer_Dog', 'N/A'):.3f}"])
        orig_rows.append(['Deer → Horse', f"{orig_data.get('ARI_Deer_Horse', 'N/A'):.3f}"])
        
        if rows:
            for i, orig_row in enumerate(orig_rows):
                if i < len(rows):
                    rows[i].append(orig_row[1])
                else:
                    rows.append([orig_row[0], "N/A", orig_row[1]])
        else:
            for orig_row in orig_rows:
                rows.append([orig_row[0], "N/A", orig_row[1]])
    
    header = ['Cases', 'vgg16']
    if original_data and 'vgg16' in original_data:
        header.append('vgg16 (Original)')
    
    table = ax.table(cellText=rows, colLabels=header, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)
    
    for i in range(len(header)):
        table[(0, i)].set_text_props(weight='bold')
        table[(0, i)].set_edgecolor('black')
        table[(0, i)].set_linewidth(1.5)
        
    for i in range(1, len(rows) + 1):
        for j in range(len(header)):
            table[(i, j)].set_text_props(color='black')
            table[(i, j)].set_edgecolor('black')
            table[(i, j)].set_linewidth(1)
        
        table[(i, 0)].set_text_props(weight='bold', color='black')
    
    plt.title('Table 2: Performance comparison of VGG16 classifier on CIFAR-10', 
              fontsize=14, fontweight='bold', pad=20)
    
    output_file = os.path.join(output_dir, 'table2_cifar_vgg16.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Table 2 saved to: {output_file}")
    return output_file

def main():
    print("="*70)
    print("Generating Table 2: Performance comparison of VGG16 classifier on CIFAR-10")
    print("="*70)
    
    output_dir = SCRIPT_DIR
    
    print("\nLoading CIFAR VGG16 data...")
    all_data = load_cifar_data()
    
    if not all_data:
        print("No CIFAR data found!")
        return
    
    print(f"\nLoaded data for {len(all_data)} case(s):")
    for case, data in all_data.items():
        print(f"  Case {case}: {len(data['max_aucs'])} replications")
    
    expected_cases = ['3-5', '4-5', '4-7']
    missing_cases = [c for c in expected_cases if c not in all_data]
    if missing_cases:
        print(f"\nMissing case files: {missing_cases}")
        print("  Expected files like: cifar_4-5_n_1000_seed_*.pkl, cifar_4-7_n_1000_seed_*.pkl")
    
    all_max_aucs = []
    for case, data in all_data.items():
        all_max_aucs.extend(data['max_aucs'])
    
    reproduced_results = {}
    reproduced_results['vgg16'] = {
        'Size': calculate_size_performance(all_max_aucs),
        'Power': calculate_power_performance(all_max_aucs),
        'ARI': {}
    }
    
    for case, data in all_data.items():
        aris = data['aris']
        if aris:
            avg_ari = np.mean(aris)
            reproduced_results['vgg16']['ARI'][case] = round(avg_ari, 3)
            print(f"  Case {case}: Average ARI = {avg_ari:.3f} (from {len(aris)} replications)")
    
    if reproduced_results['vgg16']['Size']:
        print(f"\nSize performance: {reproduced_results['vgg16']['Size']}%")
    if reproduced_results['vgg16']['Power']:
        print(f"Power performance: {reproduced_results['vgg16']['Power']}%")
    
    original_data = get_original_paper_table1()
    
    print("\nCreating table visualization...")
    create_table_visualization(reproduced_results, original_data, output_dir)
    
    print("\nTable 2 generation complete!")

if __name__ == "__main__":
    main()
