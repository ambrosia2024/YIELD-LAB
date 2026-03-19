# %% Importing libraries 
import os, sys
import joblib
import argparse
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd

from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset

sys.path.append('../process/')
from helpers import verify_parameters, select_country, seed_uniformly
from loadData import prepare_features_and_targets
from validateModel import evaluate_predictions_by_year, store_model_results, evaluate_OOD_results_from_countries

sys.path.append('../architectures/')
from sklearnLayer import generate_statistical_pipeline

import warnings
warnings.filterwarnings('ignore')

# %% Setting up the Argparser
parser = argparse.ArgumentParser(description="Trains sklearn models to perform crop yield prediction task.")
parser.add_argument('--crop', type=str, default="wheat", help="Name of the crop.")
parser.add_argument('--model', type=str, default="ridge", help="Name of the model to be trained.")
parser.add_argument('--country', type=str, default="DE", help="Country data to be trained on.")
parser.add_argument('--output_dir', type=str, default="../output/trained_models/", help="""Directory for outputs to be saved""")
parser.add_argument('--save_dir', type=str, default="../output/saved_models/", help="""Directory for models to be saved""")
parser.add_argument('--seed', type=int, default=1111, help='Random seed value')
parser.add_argument("--use_trained_model", action="store_true", help="Used to load and evaluate a trained model")

args = parser.parse_args()

# Creating output directory if it doesn't exist
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# Model file name
model_file = os.path.join(args.save_dir, "sklearn_models", f"{args.model}_pipeline.pkl")

# Verifying if the pipeline for selected crop, model, and country is implemented
verify_parameters(crop=args.crop, model=args.model, country=args.country)
if args.use_trained_model:
    if os.path.exists(model_file):
        pass
    else:
        raise Exception("Saved model doesn't exist. Consider retraining or recheck the saved diretory.")

# Managing reproducibility
seed_uniformly(seed=args.seed)

# %% Loading the data for training
# Loading the country of interest
args.country = select_country(crop=args.crop, country=args.country)

# Loading the aligned data from CY-BENCH library (targets + inputs)
df_y, dfs_x = load_dfs_crop(args.crop, countries=args.country)
# Building CY-Bench dataset object
ds = CYDataset(crop=args.crop, data_target=df_y, data_inputs=dfs_x)

# Splitting training (before 2018) and test-set (2018 and after)
years_sorted = sorted(list(ds.years))
train_years = [y for y in years_sorted if y <= 2017]
test_years  = [y for y in years_sorted if y >= 2018]
train_ds, test_ds = ds.split_on_years((train_years, test_years))

X_train, y_train, years_train = prepare_features_and_targets(train_ds)
X_test, y_test, years_test = prepare_features_and_targets(test_ds)

# %% Build Model and Training or loading a trained Pipeline
pipeline = generate_statistical_pipeline(model_name=args.model, seed=args.seed)
if not args.use_trained_model:
    Path(os.path.dirname(model_file)).mkdir(parents=True, exist_ok=True)
    pipeline.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(pipeline, model_file)
else:
    # Load a trained model file
    pipeline = joblib.load(model_file)

# %% Predicting and testing the model
y_pred = pipeline.predict(X_test)
results_by_year = evaluate_predictions_by_year(y_test, y_pred, years_test)
print("Model Performance by Year:", results_by_year)

# Storing results
_ = store_model_results(results_dict=results_by_year, model_name=args.model, country=args.country, crop=args.crop, file_path=os.path.join(args.output_dir, f"sklearn_models.csv"))

# %% Evaluating the trained model on EU countries
evaluate_OOD_results_from_countries(crop=args.crop, model_name=args.model, pipeline=pipeline, file_path=os.path.join(args.output_dir, f"sklearn_models.csv"))