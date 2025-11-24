# COMP3314 Result Visualisation

This repository contains visualization and analysis scripts for reproducing results from change point detection experiments on real and simulated data.

## Visualization Scripts

### Real Data Visualizations

#### 1. NYC Taxi (`nyc_taxi_visualization/`)
Visualizes change points detected in NYC taxi drop-off data using VGG16 features.

**Script:** `visualize_vgg16_nyc_taxi.py`

**Outputs:**
- `vgg16_nyc_taxi_timeseries.png` - Time series plot with detected change points
- `vgg16_nyc_taxi_table.png` - Comparison table of original and reproduced change points

**Usage:**
```bash
cd nyc_taxi_visualization
python3 visualize_vgg16_nyc_taxi.py
```

**Data Source:**
- `real_data/nyc_taxi/vgg16_sbs_nyc_taxi.pkl` - Change point detection results
- `real_data/nyc_taxi/heatmaps_color_numeric.pkl` - Original taxi data

#### 2. US Stocks (`us_stocks_visualization/`)
Visualizes change points detected in US stocks data.

**Script:** `visualize_us_stocks.py`

**Outputs:**
- `us_stocks_timeseries.png` - Time series plot with detected change points
- `us_stocks_table5.png` - Comparison table of original and reproduced change points

**Usage:**
```bash
cd us_stocks_visualization
python3 visualize_us_stocks.py
```

**Data Source:**
- `real_data/us_stocks/output/` - Change point detection results
- `real_data/us_stocks/us_stocks/stable_stocks.csv` - Stock data with dates

**Methods:**
- changeforest
- fnn
- gseg_wei, gseg_max, gseg_orig, gseg_gen

### Simulated Data Visualizations

#### 1. Computation Time (`simulated_data/simulated_data_visualization/`)
Average computation time (in seconds) across 1000 replications from the 'Dense Mean Change' setup with T = 1000.

**Script:** `generate_table3.py`

**Outputs:**
- `table3_computation_time.png` - Comparison of original paper and reproduced computation times

**Usage:**
```bash
cd simulated_data/simulated_data_visualization
python3 generate_table3.py
```

**Data Source:**
- Log files: `simulated_data/simulated_data_output/logs/`
- RData files: `simulated_data/result_3/output/dense_mean/`
- Methods: `Logis`, `Rf`, `Fnn`, `gseg`, `changeforest`, `Hddc`, `NODE`

#### 2. ARI and AUC Visualizations (`simulated_data/simulated_data_visualization/`)
Creates boxplot visualizations for ARI (Adjusted Rand Index) and AUC (Area Under Curve) performance.

**Script:** `visualize_ari_auc.py`

**Outputs:**
- `ari_boxplot.png` - ARI performance boxplot (p=500)
- `auc_boxplot.png` - AUC performance boxplot (p=500)

**Usage:**
```bash
cd simulated_data/simulated_data_visualization
python3 visualize_ari_auc.py
```

**Data Source:**
- `simulated_data/computational_time/output/dense_mean/`
- Methods: `changeforest`, `Logis`, `FNN`, `gseg`, `RF`
- Only includes data with p=500
- True change point: 500

**Method Order and Colors:**
1. changeforest (red)
2. Logis (blue)
3. FNN (orange)
4. RF (purple)

#### 3. Simulated Data Output Visualizations (`simulated_data/simulated_data_visualization/`)
Creates boxplot visualizations for ARI and AUC from simulated_data_output folder.

**Script:** `visualize_simulated_data_output.py`

**Outputs:**
- `simulated_output_ari_boxplot.png` - ARI performance by null type
- `simulated_output_auc_boxplot.png` - AUC performance by null type

**Usage:**
```bash
cd simulated_data/simulated_data_visualization
python3 visualize_simulated_data_output.py
```

**Data Source:**
- `simulated_data/simulated_data_output/output/`
- Organized by null types: `standard_null`, `banded_null`, `exp_null`
- Methods: `reg_logis`, `rf`, `fnn`

## Null Types

The simulated data experiments use three null types:

1. **Standard Null**: {**Z**_t_}_t_=1^T^ iid ~ *N*~p~(**0**_p_, *I*~p~)
   - Independent and identically distributed p-variate normal with zero mean and identity covariance

2. **Banded Null**: {**Z**_t_}_t_=1^T^ iid ~ *N*~p~(**0**_p_, Σ) with Σ = (*σ*~ij~)_i,j_=1^p^, *σ*~ij~ = 0.8^|i-j|^
   - p-variate normal with banded covariance structure

3. **Exponential Null**: {*Z*~tj~}_t_=1^T^ iid ~ Exp(1) for *j* = 1, ..., *p*
   - Independent exponential distributions for each component

- `simulated_data/simulated_data_output/output/`
- Organized by null types: `standard_null`, `banded_null`, `exp_null`
- Methods: `reg_logis`, `rf`, `fnn`

## Requirements

- **Python 3.x**
- **R** (with `jsonlite` package installed)
- **Python packages:**
  - numpy
  - matplotlib
  - pandas
  - pathlib

## Data File Formats

### RData Files
- Binary R data files containing simulation results
- Extracted using Rscript with jsonlite package

### Pickle Files (.pkl)
- Python binary files containing change points or results
- Loaded using Python's pickle module

### Log Files (.log)
- Text files containing runtime information
- Parsed using regex to extract "finished in X secs" patterns

## Running All Visualizations

To generate all visualizations:

```bash
# Real data
cd nyc_taxi_visualization && python3 visualize_vgg16_nyc_taxi.py
cd ../us_stocks_visualization && python3 visualize_us_stocks.py

# Simulated data
cd ../simulated_data/simulated_data_visualization
python3 generate_table3.py
python3 visualize_ari_auc.py
python3 visualize_simulated_data_output.py
```
