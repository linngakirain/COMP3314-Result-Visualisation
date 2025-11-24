# COMP3314 Result Visualisation

This repository contains visualization and analysis scripts for reproducing results from change point detection experiments on real and simulated data.

## Visualization Scripts

### Tables

#### Table 1: Average Computation Time (`simulated_data/simulated_data_visualization/`)
Average computation time (in seconds) across 1000 replications from the 'Dense Mean Change' setup with T = 1000.

**Script:** `generate_table1.py`

**Outputs:**
- `table1_computation_time.png` - Comparison of original paper and reproduced computation times

**Usage:**
```bash
cd simulated_data/simulated_data_visualization
python3 generate_table1.py
```

**Data Source:**
- Log files: `simulated_data/simulated_data_output/logs/`
- RData files: `simulated_data/result_3/output/dense_mean/`
- Methods: `Logis`, `Rf`, `Fnn`, `gseg`, `changeforest`, `Hddc`, `NODE`

#### Table 2: Performance Comparison of VGG16 Classifier on CIFAR-10 (`simulated_data/simulated_data_visualization/`)
Size and Power performance, and average ARI values for CIFAR VGG16.

**Script:** `generate_table2.py`

**Outputs:**
- `table2_cifar_vgg16.png` - Size, Power, and ARI performance table

**Usage:**
```bash
cd simulated_data/simulated_data_visualization
python3 generate_table2.py
```

**Data Source:**
- `simulated_data/cifar/vgg16/`
- Cases: Cat → Dog (3-5), Deer → Dog (4-5), Deer → Horse (4-7)

#### Table 3: Comparison of Detected Change-Points in US Stock Market Data (`us_stocks_visualization/`)
Comparison table of original and reproduced change points in US stocks data.

**Script:** `visualize_us_stocks.py`

**Outputs:**
- `table3_us_stocks.png` - Comparison table of original and reproduced change points

**Usage:**
```bash
cd us_stocks_visualization
python3 visualize_us_stocks.py
```

**Data Source:**
- `real_data/us_stocks/output/` - Change point detection results
- Methods: `changeforest`, `fnn`, `gseg_wei`, `gseg_max`, `gseg_orig`, `gseg_gen`

#### Table 4: Comparison of Detected Change-Points in NYC Taxi Data (`nyc_taxi_visualization/`)
Comparison table of original and reproduced change points in NYC taxi data.

**Script:** `visualize_vgg16_nyc_taxi.py`

**Outputs:**
- `table4_nyc_taxi.png` - Comparison table of original and reproduced change points

**Usage:**
```bash
cd nyc_taxi_visualization
python3 visualize_vgg16_nyc_taxi.py
```

**Data Source:**
- `real_data/nyc_taxi/vgg16_sbs_nyc_taxi.pkl` - Change point detection results
- `real_data/nyc_taxi/heatmaps_color_numeric.pkl` - Original taxi data

### Figures

#### Figure 2: Reproduced ARI Distributions with Dense Mean Boxplot (`simulated_data/simulated_data_visualization/`)
Boxplot visualization for ARI (Adjusted Rand Index) performance.

**Script:** `visualize_ari_auc.py`

**Outputs:**
- `figure2_ari_boxplot.png` - ARI performance boxplot (p=500)

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

#### Figure 3: AUC Performance Metrics Boxplot (`simulated_data/simulated_data_visualization/`)
Boxplot visualization for AUC (Area Under Curve) performance.

**Script:** `visualize_ari_auc.py`

**Outputs:**
- `figure3_auc_boxplot.png` - AUC performance boxplot (p=500)

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

#### Figure 4: US Stocks Time Series (`us_stocks_visualization/`)
Time series plot with detected change points in US stocks data.

**Script:** `visualize_us_stocks.py`

**Outputs:**
- `figure4_us_stocks_timeseries.png` - Time series plot with detected change points

**Usage:**
```bash
cd us_stocks_visualization
python3 visualize_us_stocks.py
```

**Data Source:**
- `real_data/us_stocks/us_stocks/stable_stocks.csv` - Stock data with dates
- `real_data/us_stocks/output/` - Change point detection results

#### Figure 5: NYC Taxi Trip Counts with Change-Points (`nyc_taxi_visualization/`)
Time series plot with detected change points in NYC taxi drop-off data.

**Script:** `visualize_vgg16_nyc_taxi.py`

**Outputs:**
- `figure5_nyc_taxi_timeseries.png` - Time series plot with detected change points

**Usage:**
```bash
cd nyc_taxi_visualization
python3 visualize_vgg16_nyc_taxi.py
```

**Data Source:**
- `real_data/nyc_taxi/heatmaps_color_numeric.pkl` - Original taxi data
- `real_data/nyc_taxi/vgg16_sbs_nyc_taxi.pkl` - Change point detection results

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
# Tables
cd simulated_data/simulated_data_visualization
python3 generate_table1.py
python3 generate_table2.py

# Figures
python3 visualize_ari_auc.py

# Real data
cd ../../us_stocks_visualization
python3 visualize_us_stocks.py

cd ../nyc_taxi_visualization
python3 visualize_vgg16_nyc_taxi.py
```
