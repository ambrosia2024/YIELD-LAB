# Crop Yield Prediction on NUTS3 region
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
git clone https://github.com/ambrosia2024/ANAND.git
mv ANAND/* ./
rm -rf ANAND
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
│   ├── train/           (from crop_yield_prediction)
│   ├── process/         (from crop_yield_prediction)
│   ├── architectures/   (from crop_yield_prediction)
|   ├── environment.yml  (from crop_yield_prediction
|   └──  (other folders and files from CY-BENCH)
└── (other files from CY-BENCH)
```

# Model training and evaluation

```
cd train/
python statistical_baselines.py --model mlp --country DE --crop wheat --seed 1111 --save_dir ../output/saved_models/ --output_dir ../output/trained_models/
```
