import os, sys
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset

import torch
import torchmetrics
import torch.nn as nn

# Custom library
from loadData import prepare_features_and_targets

def evaluate_predictions_by_year(y_true, y_pred, years, metrics=None, min_samples_per_year=2, epsilon=1e-6):
    """
    Compute regression metrics overall and per year with robustness to small sample sizes
    and zero/near-zero variance.

    Args:
        y_true: np.ndarray of true targets
        y_pred: np.ndarray of predicted targets
        years: list or np.ndarray of corresponding years
        metrics: list of metrics to compute ['r2', 'rmse', 'mape', 'normalized_rmse']
        min_samples_per_year: minimum number of samples to compute metrics per year
        epsilon: small value to avoid division by zero in normalized RMSE

    Returns:
        dict: nested dictionary {overall: {...}, year1: {...}, ...}
    """

    if metrics is None:
        metrics = ['r2', 'rmse', 'mape', 'normalized_rmse']

    EPSILON = epsilon
    results = {}

    # --- Metric functions ---
    def compute_r2_or_nan(y_t, y_p):
        if len(y_t) < min_samples_per_year or np.std(y_t) < EPSILON:
            return np.nan
        return r2_score(y_t, y_p)

    def compute_rmse(y_t, y_p):
        if len(y_t) == 0:
            return np.nan
        return np.sqrt(mean_squared_error(y_t, y_p))

    def compute_mape(y_t, y_p):
        if len(y_t) == 0:
            return np.nan
        return mean_absolute_percentage_error(y_t, y_p)

    def compute_normalized_rmse(y_t, y_p):
        if len(y_t) < min_samples_per_year:
            return compute_rmse(y_t, y_p)
        denom = np.max(y_t) - np.min(y_t)
        return compute_rmse(y_t, y_p) / (denom + EPSILON) * 100

    # --- Overall metrics ---
    results['overall'] = {}
    for metric in metrics:
        if metric == 'r2':
            results['overall']['r2'] = compute_r2_or_nan(y_true, y_pred)
        elif metric == 'rmse':
            results['overall']['rmse'] = compute_rmse(y_true, y_pred)
        elif metric == 'mape':
            results['overall']['mape'] = compute_mape(y_true, y_pred)
        elif metric == 'normalized_rmse':
            results['overall']['normalized_rmse'] = compute_normalized_rmse(y_true, y_pred)

    # --- Per-year metrics ---
    years = np.array(years)
    for year in sorted(set(years)):
        mask = np.where(years == year)[0]
        y_t = y_true[mask]
        y_p = y_pred[mask]

        results[year] = {}
        for metric in metrics:
            if metric == 'r2':
                results[year]['r2'] = compute_r2_or_nan(y_t, y_p)
            elif metric == 'rmse':
                results[year]['rmse'] = compute_rmse(y_t, y_p)
            elif metric == 'mape':
                results[year]['mape'] = compute_mape(y_t, y_p)
            elif metric == 'normalized_rmse':
                results[year]['normalized_rmse'] = compute_normalized_rmse(y_t, y_p)

    return results


def store_model_results(results_dict, model_name, country, crop, file_path="../output/myoutputs/sklearn_model_results.csv"):
    rows = []
    for year, metrics in results_dict.items():
        for metric_name, value in metrics.items():
            rows.append({
                "model": str(model_name),
                "country": str(country[0]),
                "crop": str(crop),
                "year": str(year),
                "metric": str(metric_name),
                "value": value
            })

    new_df = pd.DataFrame(rows)

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df = combined_df.drop_duplicates(subset=["model", "country", "crop", "year", "metric"])
    combined_df.to_csv(file_path, index=False)
    # print(f"Results stored successfully. Total records: {len(combined_df)}")
    return combined_df

def evaluate_OOD_results_from_countries(crop, model_name, pipeline, file_path):
    """
    Evaluates the trained model on EU countries in CY-BENCH dataset.
    """

    countries_to_evaluate = ["AT", "BE", "BG", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LV", "NL", "PL", "PT", "RO", "SE", "SK"]

    if crop == "maize":
        countries_not_of_interest = ["EE", "FI", "IE", "LV"]
    elif crop == "wheat":
        countries_not_of_interest = ["SK"]
    else:
        raise Exception("Crop can either be maize or wheat.")

    for country in countries_not_of_interest:
        countries_to_evaluate.remove(country)
    
    # for country in tqdm(countries_to_evaluate):
    pbar = tqdm(countries_to_evaluate, desc="Evaluating")
    for country in pbar:
        pbar.set_description(f"Evaluating {model_name} model on {country} country")

        df_y, dfs_x = load_dfs_crop(crop, countries=[country])
        ds = CYDataset(crop=crop, data_target=df_y, data_inputs=dfs_x)

        years_sorted = sorted(list(ds.years))
        train_years = [y for y in years_sorted if y <= 2017]
        test_years  = [y for y in years_sorted if y >= 2018]
        _, test_ds = ds.split_on_years((train_years, test_years))

        X_test, y_test, years_test = prepare_features_and_targets(test_ds)
        y_pred = pipeline.predict(X_test)

        results_by_year = evaluate_predictions_by_year(y_test, y_pred, years_test)
        _ = store_model_results(results_by_year, model_name, country, crop, file_path)


class ModelMetrics(nn.Module):
    """
    Comprehensive metrics for agricultural yield forecasting.

    Wraps torchmetrics to provide MSE, MAE, RMSE, R², MAPE, SMAPE, NRMSE.
    Prefix (train/val/test) is used for wandb logging namespacing.

    NOTE: Inherits from nn.Module so metrics are moved to device automatically
    when the parent LightningModule is moved to GPU.
    """

    def __init__(self, prefix: str = "test", include_nrmse: bool = True):
        super().__init__()
        self.prefix = prefix
        metrics_dict = {
            'mse': torchmetrics.MeanSquaredError(),
            'mae': torchmetrics.MeanAbsoluteError(),
            'r2': torchmetrics.R2Score(),
            'mape': torchmetrics.MeanAbsolutePercentageError(),
            'smape': torchmetrics.SymmetricMeanAbsolutePercentageError(),
        }
        # Only include NRMSE if requested (exclude for training since targets are normalized)
        if include_nrmse:
            metrics_dict['nrmse'] = torchmetrics.NormalizedRootMeanSquaredError(normalization='mean')
        self.metrics = torchmetrics.MetricCollection(metrics_dict)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.metrics.update(preds, targets)

    def compute(self) -> Dict:
        return self.metrics.compute()

    def reset(self):
        self.metrics.reset()

    def log_results(self, step: str = "val"):
        results = self.compute()
        print(f"\n{'-' * 60}")
        print(f"{step.upper()} METRICS ({self.prefix.upper()}):")
        print(f"  MSE:   {results['mse']:.4f}")
        print(f"  MAE:   {results['mae']:.4f}")
        print(f"  RMSE:  {torch.sqrt(results['mse']):.4f}")
        print(f"  R²:    {results['r2']:.4f}")
        # MAPE/SMAPE reported as fractions
        print(f"  MAPE:  {results['mape']:.4f}")
        print(f"  SMAPE: {results['smape']:.4f}")
        # Only print NRMSE if it exists (excluded for training metrics)
        if 'nrmse' in results:
            print(f"  NRMSE: {results['nrmse']:.4f}")
        print(f"{'-' * 60}")

def format_metrics_dict(results: Dict) -> Dict[str, float]:
    """
    Convert torchmetrics tensor results to a clean dict of floats.

    Args:
        results: Dict with tensor metrics from ModelMetrics.compute()

    Returns:
        Dict with all metrics as Python floats
    """
    return {
        'mse': float(results['mse'].item()) if 'mse' in results else None,
        'mae': float(results['mae'].item()) if 'mae' in results else None,
        'rmse': float(torch.sqrt(results['mse']).item()) if 'mse' in results else None,
        'r2': float(results['r2'].item()) if 'r2' in results else None,
        'mape': float(results['mape'].item()) if 'mape' in results else None,
        'smape': float(results['smape'].item()) if 'smape' in results else None,
        'nrmse': float(results['nrmse'].item()) if 'nrmse' in results else None,
    }

def print_metrics_table(title: str, metrics: Dict, step: str = "test"):
    """
    Print a nicely formatted table of all metrics.

    Args:
        title: Section title (e.g., "CV Fold 1 Results")
        metrics: Dict from format_metrics_dict()
        step: Step name for context
    """
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")

    if metrics.get('mse') is not None:
        print(f"  MSE:   {metrics['mse']:.4f}")
    if metrics.get('mae') is not None:
        print(f"  MAE:   {metrics['mae']:.4f}")
    if metrics.get('rmse') is not None:
        print(f"  RMSE:  {metrics['rmse']:.4f}")
    if metrics.get('r2') is not None:
        print(f"  R²:    {metrics['r2']:.4f}")
    if metrics.get('mape') is not None:
        print(f"  MAPE:  {metrics['mape']:.2f}%")
    if metrics.get('smape') is not None:
        print(f"  SMAPE: {metrics['smape']:.2f}%")
    if metrics.get('nrmse') is not None:
        print(f"  NRMSE: {metrics['nrmse']:.4f}")

    print(f"{'-' * 70}")