# A Python Toolkit for Machine Learning based Crop Yield Analysis, Prediction, and Benchmarking
This repository trains and evaluates machine learning models for crop yield prediction using the CY-BENCH dataset.

# Setup Instructions
#### Prerequisites
* Python 3.10+
* Conda package manager

#### Installation
##### 1. Create and activate a Conda environment:
```
conda create -n CYP python=3.10
conda activate CYP
```

##### 2.  Clone the CY-BENCH repository:
```
git clone https://github.com/WUR-AI/AgML-CY-BENCH.git
pip install poetry
cd AgML-CY-Bench
poetry install
```

Install the dependencies as indicated at the [CY-BENCH](https://github.com/wur-ai/agml-cy-bench) repository.

##### Step 3: Clone this repository:

```
cd cybench
git clone https://github.com/ambrosia2024/YIELD-HUB.git
mv YIELD-HUB/* ./
rm -rf YIELD-HUB
```

##### Step 4: Install project dependencies:
```
conda env update --name CYP --file environment.yml --prune
```

##### Step 5: Data Download
Download the maize and wheat datasets from [CY-BENCH data](https://zenodo.org/records/13838912) on Zenodo and place them in:

```
AgML-CY-BENCH/cybench/data/
```

The directory structure should look like:
```
AgML-CY-BENCH/
├── cybench/
│   ├── data/
│   │   ├── maize/
│   │   └── wheat/
│   ├── train/           (from YIELD-HUB)
│   ├── process/         (from YIELD-HUB)
│   ├── architectures/   (from YIELD-HUB)
|   ├── environment.yml  (from YIELD-HUB
|   └──  (other folders and files from CY-BENCH)
└── (other files from CY-BENCH)
```

# Model training and evaluation

```
cd train/
python statisticalBaselines.py --model mlp --country DE --crop wheat --seed 1111 --save_dir ../output/saved_models/ --output_dir ../output/trained_models/
python tstBaselines.py --crop maize --country NL --model_type tst --use_sota_features --use_residual_trend --use_recursive_lags --use_cwb_feature --aggregation daily --include_spatial_features
python linearBaselines.py --crop maize --country NL --model_type xlinear --use_sota_features --use_residual_trend --use_recursive_lags --use_cwb_feature --aggregation daily --include_spatial_features
```

Alternatively, feel free to use the bash script if you are working with SLURM to train all the baselines together:
```
cd train/
sbatch run_baselines.sh
```

