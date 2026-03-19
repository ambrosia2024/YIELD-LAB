import sys
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl

from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

from cybench.config import (
    GDD_BASE_TEMP, GDD_UPPER_LIMIT, LOCATION_PROPERTIES, SOIL_PROPERTIES,
    FORECAST_LEAD_TIME, KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, KEY_CROP_SEASON,
    CROP_CALENDAR_DATES
)

# Custom Classes and functions
from trendLayer import TrendModel
from modelconfig import LinearModelConfig

sys.path.append('../process/')
from validateModel import ModelMetrics
from loadData import _get_static_feature_names

# Global variables
SOTA_TEMPORAL_VARS_LIST = [
    'sin_doy', 'cos_doy',
    'sin_month', 'cos_month',
    'season_sin', 'season_cos'
]

# Remote sensing features - always included
REMOTE_SENSING_FEATURES = ['fpar', 'ndvi', 'ssm', 'rsm']
print(f"[Feature Config] SOTA Temporal vars ({len(SOTA_TEMPORAL_VARS_LIST)}): {SOTA_TEMPORAL_VARS_LIST}")


class BaseTimeSeriesModel(ABC, pl.LightningModule):
    """
    Abstract base for all time series forecasting architectures.
    """

    def __init__(self, config: LinearModelConfig, lr: float = 1e-4, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.config = config
        self.trend_model = TrendModel()
        self.feature_norm_params: Optional[Dict] = None

        use_sota = config.use_sota_features
        self.n_ts_features = (
            len(config.time_series_vars)
            + (len(SOTA_TEMPORAL_VARS_LIST) if use_sota else 0)
        )
        self.num_time_features = 0

        include_spatial = config.include_spatial_features
        lag_years = config.lag_years

        n_crop_calendar = 0
        for date_name in CROP_CALENDAR_DATES:
            if date_name in ["sos_date", "eos_date"]:
                n_crop_calendar += 2
            else:
                n_crop_calendar += 1

        self.n_static_features = (
            len(SOIL_PROPERTIES) + len(LOCATION_PROPERTIES) + n_crop_calendar
            + (2 if include_spatial else 0)
            + lag_years
        )

        print(f"[Model] TS features={self.n_ts_features}, Static features={self.n_static_features}")

        self._model_ready = False

        self.base_model = self._build_model()

        self.train_metrics = ModelMetrics(prefix="train", include_nrmse=False)
        self.val_metrics = ModelMetrics(prefix="val")
        self.test_metrics = ModelMetrics(prefix="test")

        # Prediction cache for recursive lag prediction
        self._yield_predictions_cache: Dict[Tuple[str, int], float] = {}

    @abstractmethod
    def _build_model(self) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def _get_static_feature_names(self) -> List[str]:
        return _get_static_feature_names(
            self.config.include_spatial_features,
            self.config.lag_years,
        )

    def _normalize_time_series(self, x_ts: torch.Tensor,
                                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Z-score normalise each time series feature using training statistics.
        """
        if self.feature_norm_params is None:
            if hasattr(self, 'trainer') and self.trainer is not None:
                dm_params = self.trainer.datamodule.feature_norm_params
                if dm_params is not None:
                    self.feature_norm_params = dm_params
                else:
                    raise RuntimeError("feature_norm_params not set in model or datamodule.")
            else:
                raise RuntimeError("feature_norm_params not set and no trainer available.")

        names = ([f'weather_{f}' for f in self.config.weather_features]
                 + [f'rs_{f}' for f in REMOTE_SENSING_FEATURES])
        if self.config.use_sota_features:
            names += [f'sota_{n}' for n in SOTA_TEMPORAL_VARS_LIST]

        if len(names) != x_ts.shape[2]:
            raise ValueError(f"TS feature name count ({len(names)}) != "
                             f"tensor dim ({x_ts.shape[2]})")

        x = x_ts.clone()
        for i, name in enumerate(names):
            key = f"ts_{name}"
            if key not in self.feature_norm_params:
                raise KeyError(f"Missing norm params for TS feature '{name}'")
            p = self.feature_norm_params[key]
            if p['std'] < 1e-8:
                x[:, :, i] = torch.zeros_like(x_ts[:, :, i])
            else:
                x[:, :, i] = (x_ts[:, :, i] - p['mean']) / p['std']
            x[:, :, i] = torch.nan_to_num(x[:, :, i], nan=0.0, posinf=0.0, neginf=0.0)

        if observed_mask is not None:
            mask_expanded = observed_mask.unsqueeze(-1).float()
            x = x * mask_expanded

        return x

    def _normalize_and_impute_static(self, x_static: torch.Tensor) -> torch.Tensor:
        """
        Z-score normalise static features then impute NaN → 0.0.
        """
        if self.feature_norm_params is None:
            return x_static

        names = self._get_static_feature_names()
        x = x_static.clone()
        for i, name in enumerate(names):
            if i >= x.shape[1]:
                break
            key = f"static_{name}"
            if key not in self.feature_norm_params:
                continue
            p = self.feature_norm_params[key]
            if p['std'] < 1e-8:
                x[:, i] = torch.zeros_like(x_static[:, i])
            else:
                x[:, i] = (x_static[:, i] - p['mean']) / p['std']
            x[:, i] = torch.nan_to_num(x[:, i], nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def on_train_start(self):
        """
        Fit per-location OLS trend lines and cache.
        """
        dm = self.trainer.datamodule
        train_y_orig = dm.train_ds.y.numpy() * dm.y_std + dm.y_mean

        train_items = [
            {KEY_LOC: dm.train_ds.adm_ids[i],
             KEY_YEAR: int(dm.train_ds.years[i]),
             KEY_TARGET: float(train_y_orig[i])}
            for i in range(len(train_y_orig))
        ]
        self.trend_model.fit(train_items)
        self.feature_norm_params = dm.feature_norm_params

        train_df = self.trend_model._train_df
        logging.info(f"Fitting trends for {len(train_df[KEY_LOC].unique())} locations")

        self._verify_mask_is_used()

    def _verify_mask_is_used(self):
        """Smoke test: masked-out inputs should produce different outputs than unmasked."""
        if not hasattr(self, '_model_ready') or not self._model_ready:
            logging.info(f"[{self.config.model_type}] Skipping mask verification (model not ready).")
            return

        dummy_ts = torch.randn(2, self.config.seq_len, self.n_ts_features, device=self.device)
        dummy_static = torch.zeros(2, self.n_static_features, device=self.device)
        full_mask = torch.ones(2, self.config.seq_len, dtype=torch.bool, device=self.device)
        half_mask = full_mask.clone()
        half_mask[:, self.config.seq_len // 2:] = False

        with torch.no_grad():
            out_full = self.forward(dummy_ts, dummy_static, observed_mask=full_mask)
            out_half = self.forward(dummy_ts, dummy_static, observed_mask=half_mask)

        if torch.allclose(out_full, out_half, atol=1e-4):
            logging.warning(
                f"[{self.config.model_type}] past_observed_mask appears to have NO effect "
                f"on model output."
            )
        else:
            logging.info(f"[{self.config.model_type}] Mask verification passed.")

    def _compute_batch_trends(self, adm_ids, years: torch.Tensor, dm, lats, lons) -> torch.Tensor:
        """
        Compute normalised trend estimate for each sample in a batch.
        """
        if not self.config.use_residual_trend:
            raise RuntimeError(
                "_compute_batch_trends called with use_residual_trend=False."
            )

        test_items = []
        for i, (loc, year) in enumerate(zip(adm_ids, years)):
            year_int = int(year.item()) if hasattr(year, 'item') else int(year)
            test_items.append({
                KEY_LOC: loc,
                KEY_YEAR: year_int
            })

        trend_predictions_orig = self.trend_model._predict_trend(test_items).flatten()
        trends_z = (trend_predictions_orig - dm.y_mean) / dm.y_std
        return torch.tensor(trends_z, dtype=torch.float32, device=self.device).unsqueeze(1)

    def _compute_weighted_loss(self, pred: torch.Tensor, y: torch.Tensor,
                               validity_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between predictions and targets.
        """
        return F.mse_loss(pred, y)

    def _shared_step(self, batch, metrics: ModelMetrics, loss_key: str):
        x_ts, x_static, y, years, adm_ids, lats, lons, validity_mask = batch
        dm = self.trainer.datamodule

        if self.config.use_residual_trend:
            batch_trends = self._compute_batch_trends(adm_ids, years, dm, lats, lons)
            assert batch_trends is not None
        else:
            batch_trends = None

        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        if batch_trends is not None:
            final_pred = pred + batch_trends.squeeze(-1).detach()
        else:
            final_pred = pred
        loss = self._compute_weighted_loss(final_pred, y, validity_mask)

        metrics.update(final_pred.detach(), y.detach())
        self.log(loss_key, loss, prog_bar=True)
        return loss

    def on_test_start(self):
        """Reset prediction cache at the start of testing and initialize per-year prediction storage."""
        self._yield_predictions_cache.clear()
        if self.config.use_recursive_lags and self.config.lag_years > 0:
            logging.info("[Recursive Lags] Prediction cache cleared for testing")

        # Initialize per-year prediction storage for CSV results
        dm = self.trainer.datamodule
        if hasattr(dm, '_test_years') and dm._test_years is not None:
            self._test_years = dm._test_years
            self._per_year_preds = {year: {'preds': [], 'targets': []} for year in self._test_years}
            logging.info(f"[Per-Year Metrics] Initialized storage for test years: {sorted(self._test_years)}")
        else:
            logging.warning("[Per-Year Metrics] Datamodule has no _test_years set, per-year metrics will not be computed")
            self._test_years = set()
            self._per_year_preds = {}

    def _replace_lags_with_predictions(
        self, x_static: torch.Tensor, years: torch.Tensor, adm_ids: List[str]
    ) -> torch.Tensor:
        """
        Replace observed lag yield features with cached predictions.

        Args:
            x_static: Static features [B, n_static]
            years: Years [B]
            adm_ids: Location IDs [B]

        Returns:
            Modified x_static with lag features replaced by cached predictions
        """
        x_static_modified = x_static.clone()

        # Get indices of lag features in static features
        static_names = self._get_static_feature_names()
        lag_indices = [
            i for i, name in enumerate(static_names)
            if name.startswith('lag_yield_')
        ]

        if not lag_indices:
            return x_static_modified

        # Replace each sample's lag features with cached predictions
        for sample_idx, (adm_id, year) in enumerate(zip(adm_ids, years)):
            year_int = int(year.item()) if hasattr(year, 'item') else int(year)

            # Replace each lag feature (lag_yield_1, lag_yield_2, etc.)
            for lag_offset, lag_idx in enumerate(lag_indices, start=1):
                lag_year = year_int - lag_offset
                cache_key = (adm_id, lag_year)

                if cache_key in self._yield_predictions_cache:
                    # Use cached prediction (in original scale, will be normalized later)
                    cached_pred = self._yield_predictions_cache[cache_key]
                    x_static_modified[sample_idx, lag_idx] = cached_pred
                # else: No cached prediction available, keep original (will be NaN/imputed)

        return x_static_modified

    def _cache_predictions(
        self, predictions_z: torch.Tensor, years: torch.Tensor,
        adm_ids: List[str], dm
    ):
        """
        Cache predictions in original scale for recursive lag prediction.

        Args:
            predictions_z: Predictions in z-score space [B]
            years: Years [B]
            adm_ids: Location IDs [B]
            dm: DataModule for denormalization
        """
        device = predictions_z.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)

        # Convert to original scale
        predictions_orig = predictions_z.detach() * y_std + y_mean

        for pred, year, adm_id in zip(predictions_orig, years, adm_ids):
            cache_key = (adm_id, int(year))
            self._yield_predictions_cache[cache_key] = pred.item()

    def _eval_step_with_clipping(self, batch, metrics: ModelMetrics, loss_key: str, stage: str,
                                  return_predictions: bool = False, return_orig: bool = False):
        """
        Evaluation step with clipping for physically meaningful yield predictions.

        Unlike training, this step:
        1. Denormalizes predictions and targets to original scale (tons/ha)
        2. Clips predictions to minimum of 0 (yields cannot be negative)
        3. Logs the rate at which predictions are clipped (diagnostic)
        4. Computes metrics on clipped predictions

        Args:
            batch: Input batch
            metrics: ModelMetrics instance for this stage
            loss_key: Key for logging loss (e.g., 'val_loss', 'test_loss')
            stage: 'val' or 'test' for logging clip rate
            return_predictions: If True, include predictions_z in return
            return_orig: If True, include clipped predictions and targets in return

        Returns:
            Loss tensor (computed in z-score space for consistency with training)
            or (loss, predictions_z) if return_predictions=True
            or (loss, predictions_clipped, targets_orig, years) if return_orig=True
            or (loss, predictions_z, predictions_clipped, targets_orig, years) if both True
        """
        x_ts, x_static, y_z, years, adm_ids, lats, lons, validity_mask = batch
        dm = self.trainer.datamodule

        # Compute trends if enabled and trend model has been fitted
        # (skip during sanity check validation before on_train_start)
        if self.config.use_residual_trend and self.trend_model._train_df is not None:
            batch_trends = self._compute_batch_trends(adm_ids, years, dm, lats, lons)
            assert batch_trends is not None
        else:
            batch_trends = None

        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        final_pred_z = pred + batch_trends.squeeze(-1).detach() if batch_trends is not None else pred

        loss = self._compute_weighted_loss(final_pred_z, y_z, validity_mask)

        device = final_pred_z.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)
        final_pred_orig = final_pred_z.detach() * y_std + y_mean
        y_orig = y_z.detach() * y_std + y_mean

        final_pred_clipped = torch.clamp(final_pred_orig, min=0.0)

        # Log clip rate as diagnostic (helps identify model issues)
        clip_rate = (final_pred_orig < 0.0).float().mean()
        self.log(f'{stage}/clip_rate', clip_rate, prog_bar=False)

        # Only update metrics here — never call compute() mid-epoch
        metrics.update(final_pred_clipped, y_orig)

        self.log(loss_key, loss, prog_bar=True)

        if return_orig and return_predictions:
            return loss, final_pred_z, final_pred_clipped, y_orig, years
        if return_orig:
            return loss, final_pred_clipped, y_orig, years
        if return_predictions:
            return loss, final_pred_z
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, self.train_metrics, "train_loss")

    def on_train_epoch_end(self):
        results = self.train_metrics.compute()
        self.log('train/mse', results['mse'], prog_bar=False)
        self.log('train/mae', results['mae'], prog_bar=False)
        self.log('train/r2', results['r2'], prog_bar=False)
        self.log('train/rmse', torch.sqrt(results['mse']).item(), prog_bar=False)
        self.log('train/mape', results['mape'], prog_bar=False)
        self.log('train/smape', results['smape'], prog_bar=False)
        self.train_metrics.log_results("train")
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        self._eval_step_with_clipping(batch, self.val_metrics, "val_loss", "val")

    def on_validation_epoch_end(self):
        results = self.val_metrics.compute()
        self.log('val/mse', results['mse'], prog_bar=False)
        self.log('val/mae', results['mae'], prog_bar=False)
        self.log('val/r2', results['r2'], prog_bar=False)
        self.log('val/rmse', torch.sqrt(results['mse']).item(), prog_bar=False)
        self.log('val/mape', results['mape'], prog_bar=False)
        self.log('val/smape', results['smape'], prog_bar=False)
        self.log('val/nrmse', results['nrmse'], prog_bar=False)
        self.val_metrics.log_results("val")
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        """Test step with optional recursive lag prediction and per-year accumulation."""
        if not self.config.use_recursive_lags or self.config.lag_years == 0:
            # Use standard evaluation and get predictions for per-year metrics
            loss, preds_clipped, targets, years = self._eval_step_with_clipping(
                batch, self.test_metrics, 'test_loss', stage='test', return_orig=True
            )
            self._accumulate_per_year_predictions(preds_clipped, targets, years)
            return loss

        # Recursive lag mode: modify batch to use cached predictions
        x_ts, x_static, y_z, years, adm_ids, lats, lons, validity_mask = batch
        dm = self.trainer.datamodule

        # Replace lag features with cached predictions
        x_static_modified = self._replace_lags_with_predictions(x_static, years, adm_ids)

        # Create modified batch
        modified_batch = (x_ts, x_static_modified, y_z, years, adm_ids, lats, lons, validity_mask)

        # Run evaluation step and get predictions
        loss, preds_z, preds_clipped, targets, years = self._eval_step_with_clipping(
            modified_batch, self.test_metrics, 'test_loss', stage='test',
            return_orig=True, return_predictions=True
        )

        # Cache predictions in z-score space for recursive lag prediction
        self._cache_predictions(preds_z, years, adm_ids, dm)

        # Accumulate per-year predictions for CSV results
        self._accumulate_per_year_predictions(preds_clipped, targets, years)

    def _accumulate_per_year_predictions(self, preds: torch.Tensor, targets: torch.Tensor, years: torch.Tensor):
        """Accumulate predictions and targets per year for later metrics computation."""
        if not hasattr(self, '_per_year_preds') or not self._per_year_preds:
            return

        # Convert to numpy and iterate
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        years_np = years.cpu().numpy() if isinstance(years, torch.Tensor) else years

        for pred, target, year in zip(preds_np, targets_np, years_np):
            year_int = int(year)
            if year_int in self._per_year_preds:
                self._per_year_preds[year_int]['preds'].append(float(pred))
                self._per_year_preds[year_int]['targets'].append(float(target))

    def _compute_per_year_metrics_from_preds(self) -> dict:
        """Compute per-year metrics from accumulated predictions using torchmetrics."""
        results = {}
        for year, data in self._per_year_preds.items():
            if len(data['preds']) == 0:
                continue

            # Convert to torch tensors
            preds = torch.tensor(data['preds'])
            targets = torch.tensor(data['targets'])

            # Compute metrics using torchmetrics for consistency
            mse = MeanSquaredError()
            r2 = R2Score()
            mae = MeanAbsoluteError()
            mape = MeanAbsolutePercentageError()

            mse_val = mse(preds, targets)
            r2_val = r2(preds, targets)
            mae_val = mae(preds, targets)
            mape_val = mape(preds, targets)
            rmse_val = torch.sqrt(mse_val)

            # SMAPE - manual computation
            smape_val = torch.mean(2.0 * torch.abs(preds - targets) /
                                  (torch.abs(preds) + torch.abs(targets) + 1e-8))

            # Store results with year suffix
            results[f'mse_{year}'] = mse_val.item()
            results[f'mae_{year}'] = mae_val.item()
            results[f'rmse_{year}'] = rmse_val.item()
            results[f'r2_{year}'] = r2_val.item()
            results[f'mape_{year}'] = mape_val.item()
            results[f'smape_{year}'] = smape_val.item()
            # Add nrmse for per-year metrics (consistent with FM script)
            nrmse_val = rmse_val / (targets.mean().clamp(min=1e-8))
            results[f'nrmse_{year}'] = nrmse_val.item()

        # Compute overall metrics (across all years)
        all_preds = []
        all_targets = []
        for data in self._per_year_preds.values():
            if len(data['preds']) > 0:
                all_preds.extend(data['preds'])
                all_targets.extend(data['targets'])

        if all_preds:
            all_preds_tensor = torch.tensor(all_preds)
            all_targets_tensor = torch.tensor(all_targets)

            mse_overall = MeanSquaredError()
            r2_overall = R2Score()
            mae_overall = MeanAbsoluteError()
            mape_overall = MeanAbsolutePercentageError()

            mse_val = mse_overall(all_preds_tensor, all_targets_tensor)
            r2_val = r2_overall(all_preds_tensor, all_targets_tensor)
            mae_val = mae_overall(all_preds_tensor, all_targets_tensor)
            mape_val = mape_overall(all_preds_tensor, all_targets_tensor)
            rmse_val = torch.sqrt(mse_val)
            smape_val = torch.mean(2.0 * torch.abs(all_preds_tensor - all_targets_tensor) /
                                  (torch.abs(all_preds_tensor) + torch.abs(all_targets_tensor) + 1e-8))
            nrmse_val = rmse_val / (all_targets_tensor.mean().clamp(min=1e-8))

            results['mse_overall'] = mse_val.item()
            results['mae_overall'] = mae_val.item()
            results['rmse_overall'] = rmse_val.item()
            results['r2_overall'] = r2_val.item()
            results['mape_overall'] = mape_val.item()
            results['smape_overall'] = smape_val.item()
            results['nrmse_overall'] = nrmse_val.item()

        return results

    def on_test_epoch_end(self):
        results = self.test_metrics.compute()
        self.log('test/mse', results['mse'], prog_bar=False)
        self.log('test/mae', results['mae'], prog_bar=False)
        self.log('test/r2', results['r2'], prog_bar=False)
        self.log('test/rmse', torch.sqrt(results['mse']).item(), prog_bar=False)
        self.log('test/mape', results['mape'], prog_bar=False)
        self.log('test/smape', results['smape'], prog_bar=False)
        self.log('test/nrmse', results['nrmse'], prog_bar=False)
        self.test_metrics.log_results("test")
        self.test_metrics.reset()

        # Compute per-year metrics and store on model for CSV saving
        if hasattr(self, '_per_year_preds') and self._per_year_preds:
            self._test_results_per_year = self._compute_per_year_metrics_from_preds()

    def predict(self, batch):
        """
        Generate predictions for a batch of data without updating metrics.

        This method can be called on-demand after training to get predictions
        for new data. Unlike test_step, this does not update any metrics or
        log any results - it simply returns denormalized predictions.

        Args:
            batch: Input batch tuple (x_ts, x_static, y_z, years, adm_ids, lats, lons, validity_mask)

        Returns:
            dict: Dictionary containing:
                - predictions: Predictions in original scale (tons/ha), clipped to >= 0
                - predictions_z: Predictions in z-score space (before denormalization)
                - targets: Ground truth targets in original scale (tons/ha)
                - years: Years for each sample
                - adm_ids: Administrative IDs for each sample
                - lats: Latitudes for each sample
                - lons: Longitudes for each sample
        """
        x_ts, x_static, y_z, years, adm_ids, lats, lons, validity_mask = batch
        dm = self.trainer.datamodule

        # Compute trends if enabled
        if self.config.use_residual_trend and self.trend_model._train_df is not None:
            batch_trends = self._compute_batch_trends(adm_ids, years, dm, lats, lons)
        else:
            batch_trends = None

        # Forward pass
        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        final_pred_z = pred + batch_trends.squeeze(-1).detach() if batch_trends is not None else pred

        # Denormalize to original scale
        device = final_pred_z.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)
        predictions_orig = final_pred_z.detach() * y_std + y_mean
        targets_orig = y_z.detach() * y_std + y_mean

        # Clip predictions to physically meaningful range
        predictions_clipped = torch.clamp(predictions_orig, min=0.0)

        return {
            'predictions': predictions_clipped,
            'predictions_z': final_pred_z,
            'targets': targets_orig,
            'years': years,
            'adm_ids': adm_ids,
            'lats': lats,
            'lons': lons,
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def on_fit_start(self):
        """Log model size to wandb at the start of training."""
        print(f"\n{'=' * 60}")
        print("DEBUG: on_fit_start() called - counting model parameters...")
        print(f"{'=' * 60}")

        # Get list of parameters to debug
        params_list = list(self.parameters())
        print(f"DEBUG: Number of parameter groups: {len(params_list)}")

        # Count total parameters
        total_params = sum(p.numel() for p in params_list)
        print(f"DEBUG: Total parameters counted: {total_params:,}")

        # Calculate size
        param_size = sum(p.numel() * p.element_size() for p in params_list)
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)

        print(f"DEBUG: Parameter size (bytes): {param_size:,}")
        print(f"DEBUG: Buffer size (bytes): {buffer_size:,}")
        print(f"{'=' * 60}")
        print(f"MODEL SIZE: {model_size_mb:.2f} MB")
        print(f"Total parameters: {total_params:,}")
        print(f"{'=' * 60}\n")

        # Log directly to wandb (self.log() is not allowed in on_fit_start)
        if self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({
                'model_size_mb': model_size_mb,
                'total_params': total_params
            })

# =========================================================
# LINEAR BASELINE MODELS
# =========================================================

class NLinearYieldModel(BaseTimeSeriesModel):
    """
    NLinear baseline: single linear layer per channel with last-value subtraction.

    From "Are Transformers Effective for Time Series Forecasting?"
    (Zeng et al., AAAI 2023)
    """

    def _build_model(self) -> nn.Module:
        seq_len = self.config.seq_len

        logging.info(
            f"[NLinear BUILD] seq_len={seq_len}, "
            f"n_ts_features={self.n_ts_features}, "
            f"n_static_features={self.n_static_features}"
        )

        # One linear layer maps (seq_len,) → (1,) per channel
        self.temporal_linear = nn.Linear(seq_len, 1)

        # After pooling across channels: n_ts_features scalars + static features → yield
        combined_dim = self.n_ts_features + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, 1),
        )

        logging.info(
            f"[NLinear BUILD] temporal_linear: ({seq_len} → 1), "
            f"regression_head input: {combined_dim}"
        )

        self._model_ready = True
        return nn.Identity()

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through NLinear model.
        """
        B, T, C = x_ts.shape

        # Find last valid (non-padded) value per sample per channel
        if observed_mask is not None:
            last_valid_idx = observed_mask.long().sum(dim=1) - 1
            last_valid_idx = last_valid_idx.clamp(min=0)

            idx_expanded = last_valid_idx.view(B, 1, 1).expand(B, 1, C)
            last_val = x_ts.gather(dim=1, index=idx_expanded)
        else:
            last_val = x_ts[:, -1:, :]

        # Subtract last value (distribution shift normalization)
        x_shifted = x_ts - last_val

        # Apply shared linear layer across sequence dimension per channel
        x_t = x_shifted.transpose(1, 2)
        out = self.temporal_linear(x_t)

        # Add last value back (undo distribution shift)
        last_val_t = last_val.transpose(1, 2)
        out = out + last_val_t

        # Squeeze to (B, C)
        pooled = out.squeeze(-1)

        # Concatenate with static features and predict yield
        combined = torch.cat([pooled, x_static], dim=-1)
        return self.regression_head(combined).squeeze(-1)


class DLinearYieldModel(BaseTimeSeriesModel):
    """
    DLinear baseline: decompose into trend + remainder, apply separate linear layers.

    From "Are Transformers Effective for Time Series Forecasting?"
    (Zeng et al., AAAI 2023)
    """

    KERNEL_SIZES = {
        "daily": 25,
        "weekly": 7,
        "dekad": 5,
    }

    def _build_model(self) -> nn.Module:
        seq_len = self.config.seq_len
        kernel_size = self.KERNEL_SIZES.get(self.config.aggregation, 25)

        if kernel_size % 2 == 0:
            kernel_size += 1

        logging.info(
            f"[DLinear BUILD] seq_len={seq_len}, kernel_size={kernel_size}, "
            f"n_ts_features={self.n_ts_features}, "
            f"n_static_features={self.n_static_features}"
        )

        self._kernel_size = kernel_size

        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )

        self.trend_linear = nn.Linear(seq_len, 1)
        self.remainder_linear = nn.Linear(seq_len, 1)

        combined_dim = self.n_ts_features + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, 1),
        )

        logging.info(
            f"[DLinear BUILD] trend_linear+remainder_linear: ({seq_len} → 1), "
            f"regression_head input: {combined_dim}"
        )

        self._model_ready = True
        return nn.Identity()

    def _extract_trend(
        self,
        x: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract trend component via symmetric moving average.

        When observed_mask is provided, padded positions (mask=0) are excluded
        from the moving average to avoid diluting the trend estimate with zeros.
        """
        B, C, T = x.shape
        pad = (self._kernel_size - 1) // 2

        if observed_mask is not None:
            # Convert mask to (B, C, T) format to match x
            mask_c = observed_mask.unsqueeze(1).float()  # (B, 1, T) -> (B, 1, T)

            # Pad both x and mask
            x_padded = F.pad(x, (pad, pad), mode='replicate')
            mask_padded = F.pad(mask_c, (pad, pad), mode='constant', value=0.0)

            # Compute masked moving average:
            # - AvgPool1d divides by kernel_size, so we divide both x*mask and mask
            # - The kernel_size divisions cancel: (sum(x*mask)/k) / (sum(mask)/k) = sum(x*mask)/sum(mask)
            # - This gives the correct mean over only valid (non-padded) positions
            trend_padded = self.moving_avg(x_padded * mask_padded)
            mask_sum = self.moving_avg(mask_padded).clamp(min=1e-8)

            # Normalize by actual mask sum to get true masked mean
            trend_padded = trend_padded / mask_sum
            trend = trend_padded[:, :, :T]
        else:
            # Original behavior when no mask is provided
            x_padded = F.pad(x, (pad, pad), mode='replicate')
            trend = self.moving_avg(x_padded)
            trend = trend[:, :, :T]

        return trend

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through DLinear model.
        """
        x_t = x_ts.transpose(1, 2)

        trend = self._extract_trend(x_t, observed_mask=observed_mask)
        remainder = x_t - trend

        trend_out = self.trend_linear(trend)
        remainder_out = self.remainder_linear(remainder)

        pooled = (trend_out + remainder_out).squeeze(-1)

        combined = torch.cat([pooled, x_static], dim=-1)
        return self.regression_head(combined).squeeze(-1)


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series.

    Computes per-instance, per-channel mean and std:
        mean = sum(x * mask) / sum(mask)
        std = sqrt(sum((x - mean)^2 * mask) / sum(mask) + eps)

    Note: Uses population std (dividing by N, not N-1) for consistency
    with the Reversible Instance Normalization paper.

    The affine transform (gamma, beta) is learned during training.
    """

    def __init__(self, num_features: int, eps: float = 1e-8, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(
        self,
        x: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize input using per-instance, per-channel statistics.
        """
        B, T, C = x.shape

        if observed_mask is not None:
            mask_f = observed_mask.float().unsqueeze(-1)
            valid_counts = mask_f.sum(dim=1).clamp(min=1)

            instance_mean = (x * mask_f).sum(dim=1, keepdim=True) / valid_counts.unsqueeze(1)

            sq_dev = ((x - instance_mean) ** 2 * mask_f).sum(dim=1, keepdim=True)
            instance_std = torch.sqrt(sq_dev / valid_counts.unsqueeze(1) + self.eps)
        else:
            instance_mean = x.mean(dim=1, keepdim=True)
            instance_std = x.std(dim=1, keepdim=True) + self.eps

        x_norm = (x - instance_mean) / instance_std

        if observed_mask is not None:
            x_norm = x_norm * observed_mask.float().unsqueeze(-1)

        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm, instance_mean, instance_std


class RLinearYieldModel(BaseTimeSeriesModel):
    """
    RLinear: NLinear with RevIN instance normalization.

    From "Revisiting Long-term Time Series Forecasting" (Li et al.)
    """

    def _build_model(self) -> nn.Module:
        seq_len = self.config.seq_len

        logging.info(
            f"[RLinear BUILD] seq_len={seq_len}, "
            f"n_ts_features={self.n_ts_features}, "
            f"n_static_features={self.n_static_features}"
        )

        self.revin = RevIN(
            num_features=self.n_ts_features,
            eps=1e-8,
            affine=True,
        )

        self.temporal_linear = nn.Linear(seq_len, 1)

        combined_dim = self.n_ts_features + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, 1),
        )

        logging.info(
            f"[RLinear BUILD] revin channels={self.n_ts_features} (affine=True), "
            f"temporal_linear: ({seq_len} → 1), "
            f"regression_head input: {combined_dim}"
        )

        self._model_ready = True
        return nn.Identity()

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through RLinear model.
        """
        B, T, C = x_ts.shape

        # RevIN normalize
        x_revin, instance_mean, instance_std = self.revin(x_ts, observed_mask)

        # Apply shared linear across sequence dimension per channel
        x_t = x_revin.transpose(1, 2)
        out = self.temporal_linear(x_t)

        # Squeeze to (B, C)
        pooled = out.squeeze(-1)

        # Concatenate static features and predict yield
        combined = torch.cat([pooled, x_static], dim=-1)
        return self.regression_head(combined).squeeze(-1)


class XLinearGatingBlock(nn.Module):
    """
    Shared gating block used by XLinear.
    """

    def __init__(self, input_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, input_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_mlp(x)
        out = x * gate
        out = self.dropout(out)
        return self.norm(out + x)


class XLinearYieldModel(BaseTimeSeriesModel):
    """
    XLinear adapted for crop yield regression with static features.

    From "XLinear: A Lightweight and Accurate MLP-Based Model for
    Long-Term Time Series Forecasting with Exogenous Inputs"
    (Chen et al., AAAI 2026)

    Optional RevIN normalization:
    When config.use_revIN=True, applies Reversible Instance Normalization
    to the endogenous series before embedding. This provides per-instance
    normalization (vs. global z-scoring), which can help with distribution
    shift across locations and years.
    """

    HIDDEN_SIZE = 64
    TEMPORAL_FF = 128
    CHANNEL_FF = 16
    DROPOUT = 0.1

    def _build_model(self) -> nn.Module:
        seq_len = self.config.seq_len
        n_exo = self.n_ts_features
        hidden = self.HIDDEN_SIZE
        t_ff = self.TEMPORAL_FF
        c_ff = self.CHANNEL_FF
        drop = self.DROPOUT

        logging.info(
            f"[XLinear BUILD] seq_len={seq_len}, n_exo_channels={n_exo}, "
            f"n_static={self.n_static_features}, hidden={hidden}, "
            f"temporal_ff={t_ff}, channel_ff={c_ff}"
        )

        # RevIN for endogenous series normalization (optional)
        if self.config.use_revIN:
            self.revin = RevIN(
                num_features=1,  # endogenous series is 1-dimensional
                eps=1e-8,
                affine=True,
            )
            logging.info("[XLinear BUILD] RevIN normalization enabled for endogenous series")

        # Endogenous embedding
        self.endo_embed = nn.Linear(1, hidden)

        # Learnable global token
        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.global_token, std=0.02)

        # Exogenous embedding
        self.exo_embed = nn.Linear(1, hidden)

        # Time-wise Gating Module (TGM)
        self.tgm = XLinearGatingBlock(
            input_dim=hidden,
            ff_dim=t_ff,
            dropout=drop,
        )

        # Variate-wise Gating Module (VGM)
        self.vgm = XLinearGatingBlock(
            input_dim=2 * hidden,
            ff_dim=c_ff,
            dropout=drop,
        )

        self.vgm_proj = nn.Linear(2 * hidden, hidden)

        # Prediction head
        head_input_dim = hidden + (n_exo * hidden) + self.n_static_features

        self.regression_head = nn.Sequential(
            nn.Linear(head_input_dim, head_input_dim // 2),
            nn.LayerNorm(head_input_dim // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(head_input_dim // 2, head_input_dim // 4),
            nn.LayerNorm(head_input_dim // 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(head_input_dim // 4, 1),
        )

        logging.info(
            f"[XLinear BUILD] head_input_dim={head_input_dim} "
            f"(endo_pooled={hidden} + exo_pooled={n_exo * hidden} "
            f"+ static={self.n_static_features})"
        )

        self._model_ready = True
        return nn.Identity()

    def _build_endogenous_series(
        self,
        x_static: torch.Tensor,
        x_ts: torch.Tensor,
        observed_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Construct the endogenous series from lag yield features.

        Applies RevIN normalization if enabled.
        """
        B, T, C = x_ts.shape

        if self.config.lag_years > 0:
            static_names = self._get_static_feature_names()
            lag_indices = [
                i for i, name in enumerate(static_names)
                if name.startswith('lag_yield_')
            ]

            if lag_indices:
                # Use only the most recent lag (lag_indices[0]) as the endogenous series.
                # Multiple lag years are primarily used as static features; the endogenous
                # series should represent the most recent historical yield value.
                lag_val = x_static[:, lag_indices[0]:lag_indices[0]+1]
                endo = lag_val.unsqueeze(1).expand(B, T, 1)

                if observed_mask is not None:
                    endo = endo * observed_mask.unsqueeze(-1).float()

                # Apply RevIN normalization if enabled
                if self.config.use_revIN:
                    endo, _, _ = self.revin(endo, observed_mask)

                return endo

        # Fallback: mean of all exogenous channels
        endo = x_ts.mean(dim=-1, keepdim=True)

        # Apply RevIN normalization if enabled
        if self.config.use_revIN:
            endo, _, _ = self.revin(endo, observed_mask)

        return endo

    def forward(
        self,
        x_ts: torch.Tensor,
        x_static: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        XLinear forward pass.
        """
        B, T, C = x_ts.shape

        # Construct endogenous series
        x_endo = self._build_endogenous_series(x_static, x_ts, observed_mask)

        # Embed endogenous
        h_endo = self.endo_embed(x_endo)

        # Initialize global token
        G = self.global_token.expand(B, 1, self.HIDDEN_SIZE)

        # Embed exogenous channels
        x_exo_reshaped = x_ts.permute(0, 2, 1).reshape(B * C, T, 1)
        h_exo_flat = self.exo_embed(x_exo_reshaped)
        h_exo = h_exo_flat.reshape(B, C, T, self.HIDDEN_SIZE)

        # Apply TGM to endogenous embeddings
        h_tgm = self.tgm(h_endo)

        # Update global token
        if observed_mask is not None:
            mask_f = observed_mask.float().unsqueeze(-1)
            valid_counts = mask_f.sum(dim=1).clamp(min=1)
            tgm_mean = (h_tgm * mask_f).sum(dim=1, keepdim=True)
            tgm_mean = tgm_mean / valid_counts.unsqueeze(1)
        else:
            tgm_mean = h_tgm.mean(dim=1, keepdim=True)

        G = G + tgm_mean

        # VGM: G × each exo channel
        G_expanded = G.expand(B, T, self.HIDDEN_SIZE)
        h_exo_t = h_exo.permute(0, 2, 1, 3)
        G_for_vgm = G_expanded.unsqueeze(2).expand(B, T, C, self.HIDDEN_SIZE)

        vgm_input = torch.cat([G_for_vgm, h_exo_t], dim=-1)
        vgm_flat = vgm_input.reshape(B * T * C, 2 * self.HIDDEN_SIZE)
        h_vgm_flat = self.vgm(vgm_flat)
        h_vgm_flat = self.vgm_proj(h_vgm_flat)
        h_vgm = h_vgm_flat.reshape(B, T, C, self.HIDDEN_SIZE)

        # Pool and predict
        if observed_mask is not None:
            mask_f = observed_mask.float().unsqueeze(-1)
            valid_counts = mask_f.sum(dim=1).clamp(min=1)
            endo_pooled = (h_tgm * mask_f).sum(dim=1) / valid_counts

            mask_tc = observed_mask.float().unsqueeze(-1).unsqueeze(-1)
            valid_tc = mask_tc.sum(dim=1).clamp(min=1)
            exo_pooled = (h_vgm * mask_tc).sum(dim=1) / valid_tc
        else:
            endo_pooled = h_tgm.mean(dim=1)
            exo_pooled = h_vgm.mean(dim=1)

        exo_pooled_flat = exo_pooled.reshape(B, C * self.HIDDEN_SIZE)

        combined = torch.cat([
            endo_pooled,
            exo_pooled_flat,
            x_static,
        ], dim=-1)

        return self.regression_head(combined).squeeze(-1)


# =========================================================
# MODEL FACTORY
# =========================================================

def create_model(config: LinearModelConfig) -> BaseTimeSeriesModel:
    model_map = {
        "nlinear": NLinearYieldModel,
        "dlinear": DLinearYieldModel,
        "xlinear": XLinearYieldModel,
        "rlinear": RLinearYieldModel,
    }

    if config.model_type.lower() not in model_map:
        raise ValueError(f"Unknown model_type '{config.model_type}'. "
                         f"Choose from: {list(model_map)}")
    return model_map[config.model_type.lower()](
        config, lr=config.lr, weight_decay=config.weight_decay)