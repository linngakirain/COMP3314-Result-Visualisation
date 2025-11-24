# Simulated Data Visualization

This directory contains scripts for visualizing and analyzing results from simulated data experiments.

## Scripts

### 1. `generate_table3.py`
Generates Table 3: Average computation time (in seconds) across 1000 replications from the 'Dense Mean Change' setup with T = 1000.

**Outputs:**
- `table3_computation_time.png` - Comparison of original paper and reproduced computation times

**Usage:**
```bash
python3 generate_table3.py
```

**Data Source:**
- Extracts runtime from log files in `simulated_data/simulated_data_output/logs/`
- Extracts runtime from RData files in `simulated_data/result_3/output/dense_mean/`
- Methods: `Logis`, `Rf`, `Fnn`, `gseg`, `changeforest`, `Hddc`, `NODE`

### 3. `visualize_ari_auc.py`
Creates boxplot visualizations for ARI (Adjusted Rand Index) and AUC (Area Under Curve) performance.

**Outputs:**
- `ari_boxplot.png` - ARI performance boxplot (p=500)
- `auc_boxplot.png` - AUC performance boxplot (p=500)

**Usage:**
```bash
python3 visualize_ari_auc.py
```

**Data Source:**
- Extracts ARI and AUC from `simulated_data/computational_time/output/dense_mean/`
- Methods: `changeforest`, `Logis`, `FNN`, `gseg`, `RF`
- Only includes data with p=500
- True change point: 500

**Method Order:**
1. changeforest (red)
2. Logis (blue)
3. FNN (orange)
4. gseg (green)
5. RF (purple)

### 4. `visualize_simulated_data_output.py`
Creates boxplot visualizations for ARI and AUC from simulated_data_output folder.

**Outputs:**
- `simulated_output_ari_boxplot.png` - ARI performance by null type
- `simulated_output_auc_boxplot.png` - AUC performance by null type

**Usage:**
```bash
python3 visualize_simulated_data_output.py
```

**Data Source:**
- Extracts ARI and AUC from `simulated_data/simulated_data_output/output/`
- Organized by null types: `standard_null`, `banded_null`, `exp_null`
- Methods: `reg_logis`, `rf`, `fnn`

## Null Types

1. **Standard Null**: {**Z**_t_}_t_=1^T^ iid ~ *N*~p~(**0**_p_, *I*~p~)
2. **Banded Null**: {**Z**_t_}_t_=1^T^ iid ~ *N*~p~(**0**_p_, Σ) with Σ = (*σ*~ij~)_i,j_=1^p^, *σ*~ij~ = 0.8^|i-j|^
3. **Exponential Null**: {*Z*~tj~}_t_=1^T^ iid ~ Exp(1) for *j* = 1, ..., *p*

- Organized by null types: `standard_null`, `banded_null`, `exp_null`
- Methods: `reg_logis`, `rf`, `fnn`
