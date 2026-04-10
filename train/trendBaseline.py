#!/usr/bin/env python3
"""
CY-BENCH Trend+OLS Baseline Model for Crop Yield Prediction
===========================================================

A baseline model that uses only temporal trend modeling via OLS regression,
without any neural network components. This serves as a strong baseline for
evaluating whether more complex architectures add value.

The model decomposes yield as: yield = trend(location, year) + residual
- trend captures technology drift / climate change over years
- residual is not modeled (this is trend-only prediction)

This script has the same complete infrastructure as timeSeriesLSTFLinear.py
and timeSeriesFM.py, ensuring fair comparison across architectures.

Usage:
    python timeSeriesTrend.py --crop maize --country NL --epochs 5 --aggregation daily --test_years 5 --lag_years 0 --include_spatial_features --use_cwb_feature
"""

import os
import sys
import random
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime
import uuid

# Configure module-level logger BEFORE any imports that might use it
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchmetrics
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Import data loading and utilities from timeSeriesLSTFLinear
sys.path.append('/gpfs/home4/vsaxena1/AgML-CY-Bench/cybench/benchmarking/MODIFIED')
from timeSeriesLSTFLinear import (
    DailyYieldDataset,
    DailyCYBenchSeqDataModule,
    build_daily_input_sequence,
    calculate_fixed_split,
    print_metrics_table,
)

# Import TrendModel and ModelMetrics from timeSeriesFM
sys.path.append('/gpfs/home4/vsaxena1/AgML-CY-Bench/cybench/benchmarking/MODIFIED')
from timeSeriesFM import TrendModel, ModelMetrics

# Import cybench utilities
from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset
from cybench.config import (
    KEY_LOC, KEY_YEAR, KEY_TARGET,
    SOIL_PROPERTIES, LOCATION_PROPERTIES, CROP_CALENDAR_DATES,
)

# pymannkendall is required for trend detection
try:
    import pymannkendall as trend_mk
    HAS_PYMANNKENDALL = True
except ImportError:
    HAS_PYMANNKENDALL = False
    trend_mk = None
    logging.warning("pymannkendall not installed. Trend detection will be disabled. "
                   "Install with: pip install pymannkendall")

# Set matmul precision conditionally based on GPU capability
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 8:  # Ampere or newer
        torch.set_float32_matmul_precision('high')
        logger.info(f"Enabled high matmul precision (GPU capability {capability})")
    else:
        logger.info(f"Keeping default matmul precision (GPU capability {capability} < 8.0)")
else:
    logger.info("Running on CPU, matmul precision setting has no effect")


@dataclass
class TrendModelConfig:
    """
    Configuration for TrendModel baseline.

    This is a simplified version of ModelConfig that only includes
    parameters relevant to the trend model, ensuring compatibility
    with the shared data pipeline.
    """
    # Model architecture
    model_type: str = "trend_ols"  # Identifier for this model type

    # Data settings (for compatibility with data pipeline)
    crop: str = "maize"
    country: str = "USA"
    aggregation: str = "dekad"  # daily, weekly, or dekad
    seq_len: int = 365  # Will be set based on aggregation

    # Features (for compatibility with data pipeline)
    time_series_vars: List[str] = None
    weather_features: List[str] = None
    include_spatial_features: bool = True
    lag_years: int = 0
    use_sota_features: bool = False
    use_cwb_feature: bool = False
    drop_tavg: bool = False

    # Domain feature engineering flags (all off by default)
    use_gdd: bool = False           # GDD time series channel
    use_heat_stress_days: bool = False  # Heat stress static counts
    use_rue: bool = False           # RUE index time series channel
    use_farquhar: bool = False      # Farquhar proxy time series channel

    # Normalization (not used by trend model, but kept for compatibility)
    use_revin: bool = False
    pooling_strategy: str = "mean"

    # Trend model specific
    use_trend_model: bool = True  # Always True for this model
    use_residual_trend: bool = False  # Trend model doesn't need residual trend
    use_recursive_lags: bool = False

    # Training settings (for compatibility with Lightning)
    max_epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    num_workers: int = 0
    load_checkpoint: Optional[str] = None
    test_years: int = 3
    results_dir: str = "checkpoints/results"

    def __post_init__(self):
        """Set default values and compute derived parameters."""
        if self.time_series_vars is None:
            # Default weather variables (not used by trend model but kept for compatibility)
            self.time_series_vars = [
                'PRECTOT', 'SRAD', 'T2M', 'T2M_MAX', 'T2M_MIN',
                'TS', 'RH2M', 'WS10M', 'fpar', 'ndvi'
            ]

        if self.weather_features is None:
            # Default weather features
            self.weather_features = ['tmin', 'tmax', 'prec', 'srad']
            if self.use_cwb_feature:
                self.weather_features.append('cwb')
            if not self.drop_tavg:
                self.weather_features.insert(2, 'tavg')

        # Set seq_len based on aggregation
        if self.aggregation == "daily":
            self.seq_len = 365
        elif self.aggregation == "weekly":
            self.seq_len = 52
        elif self.aggregation == "dekad":
            self.seq_len = 36

    def _compute_expected_static_features(self) -> int:
        """
        Compute the total expected static feature count from the current config.
        Matches the logic in ModelConfig from timeSeriesLSTFLinear.py.
        """
        n_soil = len(SOIL_PROPERTIES)
        n_location = len(LOCATION_PROPERTIES)
        n_crop = 0
        for date_name in CROP_CALENDAR_DATES:
            if date_name in ["sos_date", "eos_date"]:
                n_crop += 2  # sin and cos
            else:
                n_crop += 1
        n_spatial = 2 if self.include_spatial_features else 0
        n_lagged = self.lag_years

        return n_soil + n_location + n_crop + n_spatial + n_lagged


def generate_checkpoint_name(args) -> str:
    """
    Generate a descriptive checkpoint filename with all hyperparameters for Trend model.

    Args:
        args: ArgumentParser namespace with all hyperparameters

    Returns:
        Descriptive checkpoint filename without extension
    """
    base_name = f"{args.crop}_{args.country}"

    # Core hyperparameters for trend model
    hyperparams = [
        "model:trend_ols",  # Trend model always uses OLS
        f"agg:{args.aggregation}",
        f"epochs:{args.epochs}",
        f"lr:{args.lr}",
        f"wd:{args.weight_decay}",
        f"batch:{args.batch_size}",
        f"seed:{args.seed}",
    ]

    # Feature flags
    if args.use_sota_features:
        hyperparams.append("sota")
    if args.include_spatial_features:
        hyperparams.append("spatial")
    hyperparams.append(f"lag:{args.lag_years}")

    # Combine all parts
    parts = [base_name] + hyperparams
    name = "_".join(parts)

    return name


def save_trend_test_results_to_csv(
    config: TrendModelConfig,
    test_results: Dict[str, float],
    test_years: List[int],
    run_id: str,
    timestamp: str
):
    """
    Save trend model test results to CSV files with per-year metrics.

    Simplified version that only includes relevant columns for the trend model:
    - Timestamp, run_id, crop, country, model_type
    - Basic config: aggregation, lag_years, seed, batch_size, test_years
    - Per-year metrics (one column per test year)
    - Overall metric across all test years

    Args:
        config: TrendModelConfig with hyperparameters
        test_results: Dict of metrics with keys like 'mse_2020', 'mse_overall', etc.
        test_years: List of test year integers
        run_id: Unique run identifier
        timestamp: Timestamp string
    """
    import os

    os.makedirs(config.results_dir, exist_ok=True)

    # Only include relevant columns for trend model
    base_data = {
        'timestamp': timestamp,
        'run_id': run_id,
        'crop': config.crop,
        'country': config.country,
        'model_type': 'trend_ols',
        'aggregation': config.aggregation,
        'lag_years': config.lag_years,
        'seed': config.seed,
        'batch_size': config.batch_size,
        'test_years': config.test_years,
    }

    # Save each metric to a separate CSV file
    for metric in ['nrmse', 'mape', 'r2', 'rmse', 'mae', 'mse', 'smape']:
        csv_path = os.path.join(config.results_dir, f'{metric}.csv')
        year_columns = {str(year): test_results.get(f'{metric}_{year}', None) for year in test_years}
        year_columns['overall'] = test_results.get(f'{metric}_overall', None)
        row_data = {**base_data, **year_columns}

        # Check if file exists and has the correct header
        recreate_file = False
        if os.path.exists(csv_path):
            # Check if the header matches our expected format
            existing_df = pd.read_csv(csv_path, nrows=0)
            expected_columns = list(base_data.keys()) + list(year_columns.keys())
            if list(existing_df.columns) != expected_columns:
                print(f"[CSV Results] Existing CSV has incompatible header, recreating: {csv_path}")
                recreate_file = True

        if os.path.exists(csv_path) and not recreate_file:
            df = pd.read_csv(csv_path)
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
        else:
            df = pd.DataFrame([row_data])

        df.to_csv(csv_path, index=False)
        print(f"[CSV Results] Saved {metric} results to {csv_path}")


class LightningDatasetWrapper:
    """
    Wrapper to convert PyTorch Lightning batches to format expected by TrendModel.

    TrendModel expects items with: {KEY_LOC, KEY_YEAR, KEY_TARGET}
    Lightning batches provide: (x_ts, x_static, y, years, adm_ids, ...)

    IMPORTANT: This wrapper denormalizes targets to original scale before passing
    to TrendModel, ensuring OLS regression and Mann-Kendall trend detection operate
    on physically meaningful yield values (tons/ha) rather than z-scores.
    """

    def __init__(self, batches: List[Tuple], y_mean: float, y_std: float):
        """
        Initialize wrapper from list of Lightning batches.

        Args:
            batches: List of batches from Lightning DataLoader
            y_mean: Training set mean for denormalization
            y_std: Training set std for denormalization
        """
        self.batches = batches
        self.y_mean = y_mean
        self.y_std = y_std

    def __iter__(self):
        """Iterate over batches and convert to TrendModel format with denormalized targets."""
        for batch in self.batches:
            # Extract components from Lightning batch format
            x_ts, x_static, y, years, adm_ids = batch[:5]

            # Convert to TrendModel format with denormalized targets
            for i in range(len(y)):
                # Extract and denormalize target to original scale
                target_val = y[i].item() if torch.is_tensor(y[i]) else y[i]
                target_orig = target_val * self.y_std + self.y_mean

                yield {
                    KEY_LOC: adm_ids[i].item() if torch.is_tensor(adm_ids[i]) else adm_ids[i],
                    KEY_YEAR: years[i].item() if torch.is_tensor(years[i]) else years[i],
                    KEY_TARGET: target_orig,  # Original scale (tons/ha)
                }


class TrendYieldModel(pl.LightningModule):
    """
    Trend-only yield prediction model using OLS regression.

    This is a baseline model that predicts yield using only temporal trends
    estimated via OLS regression on historical yield data for each location.

    The model learns:
        yield(location, year) = trend(location, year)

    where trend is estimated using OLS regression on historical data for each
    location, with optimal window selection using Mann-Kendall significance test.

    This serves as a strong baseline to evaluate whether more complex neural
    architectures actually add value beyond simple trend extrapolation.
    """

    def __init__(self, config: TrendModelConfig, learning_rate: float = 1e-3):
        """
        Initialize TrendModel baseline.

        Args:
            config: Model configuration
            learning_rate: Learning rate (not used for trend model, kept for compatibility)
        """
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.learning_rate = learning_rate

        # Initialize trend model
        self.trend_model = TrendModel()

        # Track whether trend model has been fitted
        self._trend_fitted = False

        # Feature dimensions (for compatibility with data loaders)
        self.n_ts_features = len(config.time_series_vars) if config.time_series_vars else 10
        self.n_static_features = 0  # Will be determined from data

        # Metrics storage - use ModelMetrics for consistency with other models
        self.val_metrics = ModelMetrics(prefix="val", include_nrmse=True)
        self.test_metrics = ModelMetrics(prefix="test", include_nrmse=True)

        # Storage for test predictions
        self.test_predictions = []
        self.test_targets = []
        self.test_years = []
        self.test_adm_ids = []

        # Per-year metrics tracking (for CSV results)
        self._test_results_per_year = {}

        logging.info("[TrendModel] Initialized trend-only baseline model")

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: predict yield using only trend model.

        Args:
            x_ts: Time series features of shape (batch, seq_len, n_ts_features)
                  Not used by trend model, kept for compatibility
            x_static: Static features of shape (batch, n_static_features)
                     Not used directly, adm_ids and years extracted from batch
            observed_mask: Boolean mask for valid timesteps
                          Not used by trend model, kept for compatibility

        Returns:
            Predictions of shape (batch,) in yield space
        """
        batch_size = x_ts.shape[0]

        # Note: Actual trend predictions are computed in training/validation/test steps
        # where we have access to years and adm_ids. This forward method is for
        # compatibility only.

        return torch.zeros(batch_size, device=x_ts.device, dtype=x_ts.dtype)

    def _compute_batch_trend_predictions(self, batch: Tuple) -> torch.Tensor:
        """
        Compute trend predictions for a batch using fitted TrendModel.

        Args:
            batch: Lightning batch tuple

        Returns:
            Trend predictions as tensor

        Raises:
            RuntimeError: If TrendModel hasn't been fitted yet
            ValueError: If batch format is invalid
        """
        # CRITICAL FIX #9: Check if trend model is fitted before predicting
        if not self._trend_fitted:
            raise RuntimeError(
                "[TrendModel] Cannot make predictions: TrendModel has not been fitted. "
                "Ensure on_train_start() has been called (this should happen automatically "
                "at the start of training). If you're calling predict() directly without "
                "training, call model.trend_model.fit(dataset) first."
            )

        # CRITICAL FIX #10 & #13: Validate batch format before unpacking
        if not isinstance(batch, (tuple, list)) or len(batch) < 5:
            raise ValueError(
                f"[TrendModel] Invalid batch format. Expected tuple with at least 5 elements "
                f"(x_ts, x_static, y, years, adm_ids), got {type(batch)} with length "
                f"{len(batch) if isinstance(batch, (tuple, list)) else 'N/A'}"
            )

        try:
            x_ts, x_static, y, years, adm_ids = batch[:5]
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"[TrendModel] Failed to unpack batch elements: {e}. "
                f"Batch structure: {type(batch)} with {len(batch) if isinstance(batch, (tuple, list)) else 'unknown'} elements"
            ) from e

        # Validate tensor shapes and types
        batch_size = x_ts.shape[0] if hasattr(x_ts, 'shape') else len(y)

        # Convert batch to TrendModel format with robust error handling
        dataset_items = []
        for i in range(len(y)):
            # CRITICAL FIX #13: Handle both scalar and non-scalar tensor cases
            try:
                # Extract adm_id
                if torch.is_tensor(adm_ids[i]):
                    loc_val = adm_ids[i].item()
                elif isinstance(adm_ids[i], (int, float, str)):
                    loc_val = adm_ids[i]
                else:
                    loc_val = adm_ids[i] if i < len(adm_ids) else f"unknown_{i}"

                # Extract year
                if torch.is_tensor(years[i]):
                    year_val = years[i].item()
                elif isinstance(years[i], (int, float)):
                    year_val = int(years[i])
                else:
                    year_val = years[i] if i < len(years) else 2020  # Default year

                # Extract target
                if torch.is_tensor(y[i]):
                    target_val = y[i].item()
                elif isinstance(y[i], (int, float)):
                    target_val = float(y[i])
                else:
                    target_val = 0.0  # Default target

                dataset_items.append({
                    KEY_LOC: loc_val,
                    KEY_YEAR: int(year_val),
                    KEY_TARGET: target_val,
                })
            except (IndexError, TypeError, ValueError) as e:
                logging.warning(
                    f"[TrendModel] Failed to extract item {i} from batch, using defaults: {e}"
                )
                dataset_items.append({
                    KEY_LOC: f"sample_{i}",
                    KEY_YEAR: 2020,
                    KEY_TARGET: 0.0,
                })

        # CRITICAL FIX #11: Get predictions with error handling
        try:
            predictions, _ = self.trend_model.predict(dataset_items)
            predictions = np.array(predictions, dtype=np.float32)

            # Check for NaN or inf predictions
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                logging.warning(
                    "[TrendModel] Predictions contain NaN or Inf values, replacing with zeros"
                )
                predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

            # Convert to tensor (predictions are in original scale: tons/ha)
            pred_tensor = torch.tensor(predictions, device=x_ts.device, dtype=x_ts.dtype)

            # No softplus - let validation/test steps handle clipping to non-negative
            return pred_tensor
        except Exception as e:
            logging.error(f"[TrendModel] Failed to compute trend predictions: {e}")
            # Return zero predictions as fallback
            return torch.zeros(batch_size, device=x_ts.device, dtype=x_ts.dtype)

    def on_train_start(self) -> None:
        """
        Fit trend model on training data at the start of training.

        TrendModel needs to be fitted on the full training dataset before
        making predictions. We collect all training batches and fit once.

        IMPORTANT: Targets are denormalized to original scale (tons/ha) before
        fitting, ensuring OLS regression and Mann-Kendall trend detection operate
        on physically meaningful yield values.
        """
        if self._trend_fitted:
            return

        logging.info("[TrendModel] Fitting trend model on training data...")

        # Get normalization parameters from datamodule
        dm = self.trainer.datamodule
        y_mean = dm.y_mean
        y_std = dm.y_std
        logging.info(f"[TrendModel] Denormalizing targets with y_mean={y_mean:.4f}, y_std={y_std:.4f}")

        # Collect all training batches
        train_batches = []
        for batch in self.trainer.train_dataloader:
            train_batches.append(batch)

        # Wrap batches for TrendModel with denormalization
        train_dataset = LightningDatasetWrapper(train_batches, y_mean, y_std)

        # Fit trend model on original-scale yields
        self.trend_model.fit(train_dataset)
        self._trend_fitted = True

        logging.info("[TrendModel] Trend model fitting complete")

    def training_step(self, batch, batch_idx):
        """
        Training step: TrendModel doesn't need incremental training.

        The model is fitted once at the start in on_train_start().
        This step exists for compatibility with Lightning framework.
        """
        # Trend model doesn't use incremental training
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step with proper denormalization and clipping.

        Computes metrics in original scale (tons/ha) after:
        1. Getting trend predictions (already in original scale)
        2. Denormalizing targets from z-score space
        3. Clipping predictions to non-negative
        """
        x_ts, x_static, y_z, years, adm_ids = batch[:5]

        # Handle sanity check before on_train_start() is called
        if not self._trend_fitted:
            # Return dummy predictions during sanity check
            trend_pred = torch.zeros_like(y_z)
            loss = torch.tensor(0.0, device=y_z.device)

            self.val_metrics.update(trend_pred, y_z)

            self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
            return {
                'loss': loss,
                'preds': trend_pred.detach().cpu(),
                'targets': y_z.detach().cpu(),
            }

        # Get trend predictions (in original scale: tons/ha)
        trend_pred_orig = self._compute_batch_trend_predictions(batch)

        # Denormalize targets to original scale for fair comparison
        dm = self.trainer.datamodule
        device = trend_pred_orig.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)
        y_orig = y_z.detach() * y_std + y_mean

        # Clip predictions to non-negative (yields cannot be negative)
        trend_pred_clipped = torch.clamp(trend_pred_orig, min=0.0)

        # Compute loss in original scale
        loss = F.mse_loss(trend_pred_clipped, y_orig)

        # Update metrics in original scale
        self.val_metrics.update(trend_pred_clipped, y_orig)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return {
            'loss': loss,
            'preds': trend_pred_clipped.detach().cpu(),
            'targets': y_orig.detach().cpu(),
        }

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at end of epoch."""
        results = self.val_metrics.compute()
        self.val_metrics.log_results(step="validation")
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        """
        Test step with per-year metrics tracking in original scale.

        Computes metrics in original scale (tons/ha) after:
        1. Getting trend predictions (already in original scale)
        2. Denormalizing targets from z-score space
        3. Clipping predictions to non-negative
        """
        x_ts, x_static, y_z, years, adm_ids = batch[:5]

        # Handle case where trend model hasn't been fitted
        if not self._trend_fitted:
            raise RuntimeError(
                "[TrendModel] Cannot run test: TrendModel has not been fitted. "
                "Make sure training has completed before testing."
            )

        # Get trend predictions (in original scale: tons/ha)
        trend_pred_orig = self._compute_batch_trend_predictions(batch)

        # Denormalize targets to original scale for fair comparison
        dm = self.trainer.datamodule
        device = trend_pred_orig.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)
        y_orig = y_z.detach() * y_std + y_mean

        # Clip predictions to non-negative (yields cannot be negative)
        trend_pred_clipped = torch.clamp(trend_pred_orig, min=0.0)

        # Compute loss in original scale
        loss = F.mse_loss(trend_pred_clipped, y_orig)

        # Update metrics using ModelMetrics in original scale
        self.test_metrics.update(trend_pred_clipped, y_orig)

        # Store predictions for analysis
        self.test_predictions.append(trend_pred_clipped.detach().cpu())
        self.test_targets.append(y_orig.detach().cpu())
        self.test_years.append(years.detach().cpu() if torch.is_tensor(years) else years)
        self.test_adm_ids.append(adm_ids.detach().cpu() if torch.is_tensor(adm_ids) else adm_ids)

        # Track per-year metrics
        for i in range(len(y_orig)):
            year_val = years[i].item() if torch.is_tensor(years[i]) else years[i]
            year_val = int(year_val)

            if year_val not in self._test_results_per_year:
                self._test_results_per_year[year_val] = {
                    'preds': [],
                    'targets': [],
                }

            # Convert to numpy for easier computation (both in original scale)
            pred_val = trend_pred_clipped[i].item() if torch.is_tensor(trend_pred_clipped[i]) else trend_pred_clipped[i]
            target_val = y_orig[i].item() if torch.is_tensor(y_orig[i]) else y_orig[i]

            self._test_results_per_year[year_val]['preds'].append(pred_val)
            self._test_results_per_year[year_val]['targets'].append(target_val)

        # Log metrics
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test/mse', loss, prog_bar=False, on_step=False, on_epoch=True)

        return {
            'loss': loss,
            'preds': trend_pred_clipped.detach().cpu(),
            'targets': y_orig.detach().cpu(),
        }

    def on_test_epoch_end(self):
        """
        Compute and log test metrics at end of test epoch, including per-year metrics.
        """
        # CRITICAL FIX #12: Check if we have any predictions before concatenating
        if not self.test_predictions:
            logging.warning("[TrendModel] No test predictions accumulated, skipping metrics computation")
            return

        # Compute final metrics using ModelMetrics
        self.test_metrics.log_results(step="test")
        results = self.test_metrics.compute()
        self.test_metrics.reset()

        # Compute per-year metrics
        for year, data in self._test_results_per_year.items():
            if isinstance(data, dict) and 'preds' in data and len(data['preds']) > 0:
                preds = np.array(data['preds'])
                targets = np.array(data['targets'])

                # Compute metrics for this year
                mse = float(np.mean((preds - targets) ** 2))
                mae = float(np.mean(np.abs(preds - targets)))
                rmse = float(np.sqrt(mse))

                # R² score
                ss_res = float(np.sum((targets - preds) ** 2))
                ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
                r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

                # Additional metrics
                mape = float(np.mean(np.abs((targets - preds) / (targets + 1e-8)))) * 100
                smape = float(np.mean(2.0 * np.abs(preds - targets) / (np.abs(preds) + np.abs(targets) + 1e-8))) * 100
                nrmse = float(rmse / (np.mean(targets) + 1e-8))

                # Store per-year metrics with proper keys for CSV saver
                self._test_results_per_year[year]['mse'] = mse
                self._test_results_per_year[year]['mae'] = mae
                self._test_results_per_year[year]['rmse'] = rmse
                self._test_results_per_year[year]['r2'] = r2
                self._test_results_per_year[year]['mape'] = mape
                self._test_results_per_year[year]['smape'] = smape
                self._test_results_per_year[year]['nrmse'] = nrmse

        # Compute overall metrics
        all_preds = []
        all_targets = []
        for year, data in self._test_results_per_year.items():
            if isinstance(data, dict) and 'preds' in data:
                all_preds.extend(data['preds'])
                all_targets.extend(data['targets'])

        if all_preds:
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)

            overall_mse = float(np.mean((all_preds - all_targets) ** 2))
            overall_mae = float(np.mean(np.abs(all_preds - all_targets)))
            overall_rmse = float(np.sqrt(overall_mse))

            ss_res = float(np.sum((all_targets - all_preds) ** 2))
            ss_tot = float(np.sum((all_targets - np.mean(all_targets)) ** 2))
            overall_r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0

            overall_mape = float(np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-8)))) * 100
            overall_smape = float(np.mean(2.0 * np.abs(all_preds - all_targets) / (np.abs(all_preds) + np.abs(all_targets) + 1e-8))) * 100
            overall_nrmse = float(overall_rmse / (np.mean(all_targets) + 1e-8))

            self._test_results_per_year['mse_overall'] = overall_mse
            self._test_results_per_year['mae_overall'] = overall_mae
            self._test_results_per_year['rmse_overall'] = overall_rmse
            self._test_results_per_year['r2_overall'] = overall_r2
            self._test_results_per_year['mape_overall'] = overall_mape
            self._test_results_per_year['smape_overall'] = overall_smape
            self._test_results_per_year['nrmse_overall'] = overall_nrmse

        # Concatenate all predictions for saving with error handling
        try:
            # Handle both tensor and list inputs
            if all(isinstance(x, torch.Tensor) for x in self.test_predictions):
                all_preds = torch.cat(self.test_predictions)
            else:
                # Mixed types - convert to numpy first
                all_preds = np.concatenate([x.cpu().numpy() if torch.is_tensor(x) else x
                                           for x in self.test_predictions])

            if all(isinstance(x, torch.Tensor) for x in self.test_targets):
                all_targets = torch.cat(self.test_targets)
            else:
                all_targets = np.concatenate([x.cpu().numpy() if torch.is_tensor(x) else x
                                              for x in self.test_targets])

            # Handle years and adm_ids with mixed types
            if all(torch.is_tensor(x) for x in self.test_years):
                all_years = torch.cat(self.test_years)
            else:
                all_years = np.concatenate([x.cpu().numpy() if torch.is_tensor(x) else np.array([x])
                                           for x in self.test_years])

            if all(torch.is_tensor(x) for x in self.test_adm_ids):
                all_adm_ids = torch.cat(self.test_adm_ids)
            else:
                all_adm_ids = np.concatenate([x.cpu().numpy() if torch.is_tensor(x) else np.array([x])
                                              for x in self.test_adm_ids])

            # Store for later access
            self.test_results = {
                'preds': all_preds.numpy() if torch.is_tensor(all_preds) else all_preds,
                'targets': all_targets.numpy() if torch.is_tensor(all_targets) else all_targets,
                'years': all_years.numpy() if torch.is_tensor(all_years) else all_years,
                'adm_ids': all_adm_ids.numpy() if torch.is_tensor(all_adm_ids) else all_adm_ids,
            }

            logging.info(f"[TrendModel] Stored {len(all_preds)} test predictions for analysis")

        except Exception as e:
            logging.error(f"[TrendModel] Error concatenating test results: {e}")
            self.test_results = None

        # Clear storage
        self.test_predictions = []
        self.test_targets = []
        self.test_years = []
        self.test_adm_ids = []

    def configure_optimizers(self):
        """
        Configure optimizers.

        Trend model has no trainable parameters, so no optimizer needed.
        This method exists for compatibility with Lightning framework.
        """
        return None


def create_trend_model(config: TrendModelConfig) -> 'TrendYieldModel':
    """
    Create a TrendYieldModel instance.

    Args:
        config: Model configuration

    Returns:
        Configured TrendYieldModel instance
    """
    return TrendYieldModel(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CY-BENCH Trend+OLS Baseline Model for Crop Yield Prediction")
    parser.add_argument('--crop', default="maize",
                        help='Crop to train on')
    parser.add_argument('--country', default="USA",
                        help='Country code (e.g., USA, NL, CN)')
    parser.add_argument('--aggregation', default="dekad",
                        choices=['daily', 'weekly', 'dekad'],
                        help='Temporal aggregation (default: dekad)')
    parser.add_argument('--use_sota_features', action='store_true',
                        help='Use SOTA temporal features (Fourier encoding)')
    parser.add_argument('--include_spatial_features', action='store_true',
                        help='Include spatial features (lat, lon)')
    parser.add_argument('--lag_years', type=int, default=0, choices=[0, 1, 2],
                        help='Number of lagged yield years (default: 0 for trend model)')
    parser.add_argument('--load_checkpoint', default=None,
                        help='Path to checkpoint to load for evaluation')
    parser.add_argument('--save_checkpoint_dir', default='checkpoints-trend',
                        help='Directory to save model checkpoints')
    parser.add_argument('--results_dir', default='checkpoints/results',
                        help='Directory to save CSV results')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum epochs (default: 50, but trend model finishes in 1)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (not used by trend model, for compatibility)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (not used by trend model, for compatibility)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (default: 0 for HPC compatibility)')
    parser.add_argument('--test_years', type=int, default=3,
                        help='Number of years for final test set (default: 3)')
    parser.add_argument('--use_cwb_feature', action='store_true',
                        help='Include crop water balance (cwb) as a feature')
    parser.add_argument('--drop_tavg', action='store_true',
                        help='Drop tavg feature')
    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate unique run identifier and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]

    print(f"\n{'=' * 70}")
    print(f"CY-BENCH  |  TREND+OLS  |  {args.crop}-{args.country}  "
          f"|  {args.aggregation.upper()}")
    print(f"  SOTA={args.use_sota_features}  Spatial={args.include_spatial_features}  "
          f"Lag={args.lag_years}")
    print(f"  TestYears={args.test_years}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  seed={args.seed}")
    print(f"{'=' * 70}\n")

    # Create configuration
    config = TrendModelConfig(
        crop=args.crop,
        country=args.country,
        aggregation=args.aggregation,
        use_sota_features=args.use_sota_features,
        include_spatial_features=args.include_spatial_features,
        lag_years=args.lag_years,
        load_checkpoint=args.load_checkpoint,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        test_years=args.test_years,
        use_cwb_feature=args.use_cwb_feature,
        drop_tavg=args.drop_tavg,
        results_dir=args.results_dir,
    )

    print(f"[Feature Config] Weather features: {config.weather_features}")
    print(f"[Feature Config] Total time series vars ({len(config.time_series_vars)}): {config.time_series_vars}")

    # Create checkpoint directory
    os.makedirs(args.save_checkpoint_dir, exist_ok=True)
    print(f"\n[Checkpoint Config]")
    print(f"  Save directory: {args.save_checkpoint_dir}")
    if args.load_checkpoint:
        print(f"  Load checkpoint: {args.load_checkpoint}")

    # Get available years
    df_y, dfs_x = load_dfs_crop(config.crop, [config.country])
    if df_y is None or len(df_y) == 0:
        print(f"[ERROR] No data for {config.crop}-{config.country}")
        sys.exit(1)

    ds = CYDataset(config.crop, df_y, dfs_x)
    all_years = sorted(set([ds[i][KEY_YEAR] for i in range(len(ds))]))
    print(f"[Data] Available years: {all_years}")

    # Calculate fixed train/val/test split
    fixed_splits = calculate_fixed_split(
        all_years,
        test_years=args.test_years,
        val_years=2
    )

    print(f"\n[Split Config - Fixed]")
    print(f"  Total years: {fixed_splits['total_years']}")
    print(f"  Train years ({len(fixed_splits['train_years'])}): {sorted(fixed_splits['train_years'])}")
    print(f"  Val years ({len(fixed_splits['val_years'])}): {sorted(fixed_splits['val_years'])}")
    print(f"  Test years ({len(fixed_splits['test_years'])}): {sorted(fixed_splits['test_years'])}")

    # PHASE 3: Final Model Training and Evaluation
    print(f"\n{'=' * 70}")
    print(f"PHASE 3: Final Model Training and Evaluation")
    print(f"{'=' * 70}\n")

    # Create datamodule for final model
    dm_final = DailyCYBenchSeqDataModule(config)
    dm_final.setup(
        train_years=fixed_splits['train_years'],
        val_years=fixed_splits['val_years'],
        test_years=fixed_splits['test_years']
    )

    # Create final trend model
    model_final = create_trend_model(config)

    # WandB logger for final model
    try:
        wandb_logger = WandbLogger(
            project="CYBENCH-LSTF-AAAI2027",
            name=f"trend_ols-{args.crop}-{args.country}",
            config=vars(args),
            group=f"{args.crop}-{args.country}"
        )
        loggers = [wandb_logger]
    except Exception as e:
        print(f"[WandB Warning] Could not initialise WandB logger: {e}")
        from lightning.pytorch.loggers import CSVLogger
        loggers = [CSVLogger("logs/", name="cybench-trend")]

    # Setup callbacks
    final_callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=True),
        ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            dirpath=args.save_checkpoint_dir,
            filename=f'{generate_checkpoint_name(args)}_{{epoch:02d}}_{{val_loss:.4f}}',
        ),
    ]

    # Create trainer
    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=final_callbacks,
        logger=loggers,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    print("\nTraining final model...")
    print("[TrendModel] Note: Trend model fits in one epoch, additional epochs are no-ops")
    trainer.fit(model_final, dm_final)

    print("\nEvaluating final model...")
    test_results = trainer.test(model_final, dm_final, ckpt_path="best")
    if test_results:
        r = test_results[0]
        final_metrics = {
            'mse': r.get('test_loss', r.get('test/mse')),  # TrendModel logs as test_loss
            'mae': r.get('test_mae', r.get('test/mae')),
            'rmse': np.sqrt(r.get('test_loss', r.get('test/mse', 0))),
            'r2': r.get('test_r2', r.get('test/r2')),
        }
    else:
        final_metrics = {}

    # Save test results to CSV
    print(f"\n[CSV Results] Retrieving test results...")

    # Get per-year metrics if available
    if hasattr(model_final, '_test_results_per_year') and model_final._test_results_per_year:
        # Convert internal format to CSV format
        # Internal: {2016: {'mse': 1.2, 'mae': 0.8, ...}, 'mse_overall': 1.5, ...}
        # CSV format: {'mse_2016': 1.2, 'mse_overall': 1.5, 'mae_2016': 0.8, ...}
        per_year_metrics = {}

        for year, metrics in model_final._test_results_per_year.items():
            if isinstance(year, int) and isinstance(metrics, dict):
                # Per-year metrics
                for metric_name, metric_value in metrics.items():
                    if metric_name not in ['preds', 'targets']:
                        per_year_metrics[f'{metric_name}_{year}'] = metric_value
            elif isinstance(year, str) and year.endswith('_overall'):
                # Overall metrics
                per_year_metrics[year] = metrics
    else:
        print(f"[CSV Results] Warning: No per-year metrics found on model.")
        # Convert overall metrics to per-year format
        per_year_metrics = {}
        for metric, value in final_metrics.items():
            if value is not None:
                per_year_metrics[f'{metric}_overall'] = value

    # Log per-year metrics to console
    print(f"\n[CSV Results] Test Metrics:")
    if per_year_metrics:
        for year in sorted(fixed_splits['test_years']):
            print(f"  Year {year}:")
            for metric in ['mse', 'mae', 'r2']:
                key = f'{metric}_{year}'
                if key in per_year_metrics:
                    print(f"    {metric.upper()}: {per_year_metrics[key]:.4f}")

        # Log overall metrics
        if 'mse_overall' in per_year_metrics:
            print(f"\n  Overall:")
            for metric in ['mse', 'mae', 'r2']:
                key = f'{metric}_overall'
                if key in per_year_metrics:
                    print(f"    {metric.upper()}: {per_year_metrics[key]:.4f}")

    # Save to CSV
    save_trend_test_results_to_csv(
        config=config,
        test_results=per_year_metrics,
        test_years=fixed_splits['test_years'],
        run_id=run_id,
        timestamp=timestamp
    )

    # Log metrics to wandb
    try:
        import wandb
        if wandb.run is not None:
            # Calculate model size in MB
            model_size_mb = sum(p.numel() * p.element_size() for p in model_final.parameters()) / (1024 * 1024)

            # Calculate total parameters
            total_params = sum(p.numel() for p in model_final.parameters())

            # Log overall test metrics
            overall_metrics = {}
            for metric in ['mse', 'mae', 'rmse', 'r2', 'smape', 'nrmse']:
                key = f'{metric}_overall'
                if key in per_year_metrics:
                    overall_metrics[f'test/{metric}'] = per_year_metrics[key]

            # Add model metadata
            overall_metrics['model_type'] = config.model_type  # "trend_ols"
            overall_metrics['model_size_mb'] = model_size_mb
            overall_metrics['total_params'] = total_params

            if overall_metrics:
                wandb.log(overall_metrics)
                print(f"\n[WandB] Logged overall test metrics: {list(overall_metrics.keys())}")
    except ImportError:
        pass  # wandb not installed

    # Print split summary
    print(f"\n{'=' * 70}")
    print(f"SPLIT SUMMARY: {args.crop}-{args.country}")
    print(f"{'=' * 70}")
    print(f"\n  Available years ({len(all_years)}): {all_years}")
    print(f"  Train years ({len(fixed_splits['train_years'])}): {sorted(fixed_splits['train_years'])}")
    print(f"  Val years ({len(fixed_splits['val_years'])}): {sorted(fixed_splits['val_years'])}")
    print(f"  Test years ({len(fixed_splits['test_years'])}): {sorted(fixed_splits['test_years'])}")

    # Print final results
    print_metrics_table(
        f"FINAL RESULTS: {args.crop}-{args.country}",
        final_metrics
    )

    # Print experiment completion message
    print(f"\n{'=' * 70}")
    print(f"Experiment complete: {args.crop}-{args.country}")
    print(f"  Model: Trend+OLS Baseline")
    print(f"  Parameters: {sum(p.numel() for p in model_final.parameters()):,} "
          f"(all trainable parameters are from normalization layers)")
    print(f"{'=' * 70}\n")
