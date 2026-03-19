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

from transformers import (
    AutoformerModel as HFAutoformerModel,
    AutoformerConfig,
    PatchTSTModel as HFPatchTSTModel, PatchTSTConfig,
    InformerModel as HFInformerModel, InformerConfig,
    TimeSeriesTransformerModel as HFTimeSeriesTransformerModel, TimeSeriesTransformerConfig,
)

# Only for TSMixer
TimeSeriesMixerModel = None
TimeSeriesMixerForPrediction = None
TimeSeriesMixerConfig = None
try:
    # Try PatchTSMixer first (newer name)
    from transformers.models.patchtsmixer.modeling_patchtsmixer import (
        PatchTSMixerModel as TimeSeriesMixerModel,
        PatchTSMixerForPrediction as TimeSeriesMixerForPrediction,
        PatchTSMixerConfig as TimeSeriesMixerConfig,
    )
except ImportError:
    try:
        # Try time_series_transformer module (older location)
        from transformers.models.time_series_transformer import (
            TimeSeriesMixerModel,
            TimeSeriesMixerForPrediction,
            TimeSeriesMixerConfig,
        )
    except ImportError:
        try:
            # Try direct import from transformers
            from transformers import (
                PatchTSMixerModel as TimeSeriesMixerModel,
                PatchTSMixerForPrediction as TimeSeriesMixerForPrediction,
                PatchTSMixerConfig as TimeSeriesMixerConfig,
            )
        except ImportError:
            try:
                # Try TimeSeriesMixer name directly
                from transformers import (
                    TimeSeriesMixerModel,
                    TimeSeriesMixerForPrediction,
                    TimeSeriesMixerConfig,
                )
            except ImportError:
                pass  # Will handle gracefully in model factory

if TimeSeriesMixerForPrediction is None:
    logging.warning("TSMixer/PatchTSMixer not available in this transformers version. "
                   "'tsmixer' model_type will raise an error at runtime.")

# Custom Classes and functions
from trendLayer import TrendModel
from modelconfig import TSTModelConfig

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

    Provides:
    - Feature normalisation (z-score, from training stats)
    - Residual trend learning (per-location OLS, fitted in on_train_start)
    - Weighted loss computation (per-sample validity weighting)
    - Shared train / val / test step logic

    Subclasses must implement:
    - _build_model() -> nn.Module
    - forward(x_ts, x_static) -> Tensor  (shape: batch,)
    """

    def __init__(self, config: TSTModelConfig, lr: float = 1e-4, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.config = config
        self.trend_model = TrendModel()
        # NOTE: loc_trend_params removed - TrendModel handles trend prediction internally
        self.feature_norm_params: Optional[Dict] = None

        # Use config.time_series_vars property instead of global WEATHER_FEATURES
        # This ensures feature count matches the actual features being extracted
        use_sota = config.use_sota_features
        self.n_ts_features = (
            len(config.time_series_vars)
            + (len(SOTA_TEMPORAL_VARS_LIST) if use_sota else 0)
        )
        # When using SOTA features, they're passed through past_values, not as time features
        # This prevents double-counting: num_time_features should be 0 in that case
        self.num_time_features = 0  # No separate time feature embedding needed

        # Static feature count — must match _compute_expected_static_features()
        include_spatial = config.include_spatial_features
        lag_years = config.lag_years

        # Compute n_crop_calendar dynamically from CROP_CALENDAR_DATES
        # using the same cyclic-encoding logic as _compute_expected_static_features().
        # Previously hardcoded to 6, but actual CROP_CALENDAR_DATES may have fewer items.
        # This ensures n_static_features always matches what the DataModule validates.
        n_crop_calendar = 0
        for date_name in CROP_CALENDAR_DATES:
            if date_name in ["sos_date", "eos_date"]:
                n_crop_calendar += 2  # sin and cos for cyclic encoding
            else:
                n_crop_calendar += 1

        self.n_static_features = (
            len(SOIL_PROPERTIES) + len(LOCATION_PROPERTIES) + n_crop_calendar
            + (2 if include_spatial else 0)
            + lag_years
        )

        print(f"[Model] TS features={self.n_ts_features}, Static features={self.n_static_features}")

        # Flag for tracking model build completion (used by _verify_mask_is_used)
        self._model_ready = False

        #  _build_model() is the correct abstract method name.
        # The previous name _extract_static_features_build_model was a copy-paste
        # error that silently broke ABC enforcement — subclasses implementing
        # _build_model() were never actually required to by the base class.
        self.base_model = self._build_model()

        # NOTE: self.criterion (nn.MSELoss) is intentionally absent.
        # Training steps use _compute_weighted_loss() which calls
        # F.mse_loss(reduction='none') to get per-sample losses before weighting.
        # Exclude NRMSE from training metrics since targets are in z-score space (mean≈0 causes division issues)
        self.train_metrics = ModelMetrics(prefix="train", include_nrmse=False)
        self.val_metrics = ModelMetrics(prefix="val")
        self.test_metrics = ModelMetrics(prefix="test")

    # -- Abstract interface --------------------------------------------------

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """
        Instantiate and return the underlying HuggingFace model.

        Returns:
            Configured HuggingFace model instance
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x_ts: Time series features of shape (batch, seq_len, n_ts_features)
            x_static: Static features of shape (batch, n_static_features)
            observed_mask: Boolean mask of shape (batch, seq_len) indicating valid timesteps

        Returns:
            Predictions of shape (batch,) in normalised z-score space
        """
        raise NotImplementedError

    # -- Static feature name helper -----------------------------------------

    def _get_static_feature_names(self) -> List[str]:
        """
        Thin wrapper so _normalize_and_impute_static() can look up norm params.

         Previously this method only existed on DailyCYBenchSeqDataModule.
        _normalize_and_impute_static() called self._get_static_feature_names(),
        raising AttributeError on the very first training batch. Adding it here
        (delegating to the shared module-level function) fixes the crash without
        duplicating the logic.
        """
        return _get_static_feature_names(
            self.config.include_spatial_features,
            self.config.lag_years,
        )

    # -- Hidden state extraction and pooling helpers -------------------------

    def _extract_hidden_state(self, outputs) -> torch.Tensor:
        """
        Extract last hidden state from HuggingFace model outputs.

        Handles different output formats across model architectures by trying
        common attribute names in order of preference.

        Args:
            outputs: HuggingFace model output object

        Returns:
            Hidden state tensor

        Raises:
            ValueError: If no hidden state can be found
        """
        # Try single tensor attributes first
        for attr in ("encoder_last_hidden_state", "last_hidden_state"):
            val = getattr(outputs, attr, None)
            if val is not None:
                return val

        # encoder_hidden_states is a tuple — take the last layer
        val = getattr(outputs, "encoder_hidden_states", None)
        if val is not None:
            return val[-1]

        # No hidden state found — provide helpful error message
        tensor_attrs = [a for a in dir(outputs) if hasattr(getattr(outputs, a, None), 'shape')]
        raise ValueError(
            f"Could not extract hidden state from model outputs. "
            f"Available tensor attributes: {tensor_attrs}"
        )

    def _pool_hidden_state(self, h: torch.Tensor) -> torch.Tensor:
        """
        Pool hidden state to (batch, features) regardless of input shape.

        Handles different hidden state formats across model architectures:
          - (B, seq_len, d_model)          → mean over seq_len → (B, d_model)
          - (B, n_channels, n_patches, d)  → mean over patches, flatten channels → (B, n_channels * d)
          - (B, d_model)                   → already pooled → (B, d_model)

        Args:
            h: Hidden state tensor with shape (B, ...) where last dim is features

        Returns:
            Pooled tensor with shape (B, pooled_dim)

        Raises:
            ValueError: If tensor has unexpected number of dimensions
        """
        if h.dim() == 2:
            # Already pooled: (B, d_model)
            return h
        elif h.dim() == 3:
            # Standard transformer output: (B, seq_len, d_model)
            # Pool over sequence length
            return h.mean(dim=1)  # (B, d_model)
        elif h.dim() == 4:
            # PatchTST/TSMixer multivariate output: (B, n_channels, n_patches, d_model)
            # Pool over patches first, then flatten channels and d_model
            h = h.mean(dim=2)           # (B, n_channels, d_model)
            B = h.shape[0]
            return h.reshape(B, -1)     # (B, n_channels * d_model)
        else:
            raise ValueError(
                f"Unexpected hidden state shape: {h.shape} "
                f"(expected 2D, 3D, or 4D tensor, got {h.dim()}D)"
            )

    # -- Normalisation -------------------------------------------------------

    def _normalize_time_series(self, x_ts: torch.Tensor,
                                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Z-score normalise each time series feature using training statistics.
        Raises RuntimeError / KeyError early if params are missing — intentional,
        to catch silent data pipeline failures before they corrupt training.

        Re-zero padded positions after normalization to prevent spurious signal
        from padded zeros being normalized to (0 - mean)/std which is non-zero.
        """
        # During sanity check, on_train_start() hasn't been called yet
        # Try to get feature_norm_params from datamodule
        if self.feature_norm_params is None:
            if hasattr(self, 'trainer') and self.trainer is not None:
                dm_params = self.trainer.datamodule.feature_norm_params
                if dm_params is not None:
                    self.feature_norm_params = dm_params
                else:
                    raise RuntimeError("feature_norm_params not set in model or datamodule. "
                                       "Ensure datamodule.setup() has been called.")
            else:
                raise RuntimeError("feature_norm_params not set and no trainer available.")

        # Use config.weather_features instead of global WEATHER_FEATURES
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
            # Protect against zero/near-zero std to prevent inf/NaN
            if p['std'] < 1e-8:
                # Feature has no variance - set to 0 (mean in z-score space)
                x[:, :, i] = torch.zeros_like(x_ts[:, :, i])
            else:
                x[:, :, i] = (x_ts[:, :, i] - p['mean']) / p['std']
            # Handle both NaN AND inf (can be produced by division)
            x[:, :, i] = torch.nan_to_num(x[:, :, i], nan=0.0, posinf=0.0, neginf=0.0)

        # Re-zero padded positions AFTER normalization
        # Without this, padded zeros become (0 - mean)/std which is non-zero
        # and participates in attention computation as spurious signal
        if observed_mask is not None:
            # observed_mask: (batch, seq_len), x: (batch, seq_len, features)
            mask_expanded = observed_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            x = x * mask_expanded  # Zero out padded positions

        return x

    def _normalize_and_impute_static(self, x_static: torch.Tensor) -> torch.Tensor:
        """
        Z-score normalise static features then impute NaN → 0.0.

        ORDER IS CRITICAL — normalise first, impute second:
          1. z = (x - μ) / σ   →  puts data in z-score space
          2. NaN → 0.0          →  0.0 IS the mean in z-score space

        Reversing the order would impute NaN to 0.0 in original space, which
        normalises to (0 - μ)/σ — typically far below the mean, not at it.
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
            # Protect against zero/near-zero std to prevent inf/NaN
            if p['std'] < 1e-8:
                # Feature has no variance - skip normalization, set to 0 (mean in z-score space)
                x[:, i] = torch.zeros_like(x_static[:, i])
            else:
                x[:, i] = (x_static[:, i] - p['mean']) / p['std']
            # Handle BOTH NaN AND inf (from division by near-zero std)
            x[:, i] = torch.nan_to_num(x[:, i], nan=0.0, posinf=0.0, neginf=0.0)
        return x

    # -- Trend model ---------------------------------------------------------

    def on_train_start(self):
        """
        Fit per-location OLS trend lines and cache (slope, intercept).

        Trend decomposition: yield = trend(location, year) + residual
        The model learns residuals; trend is added back at inference.
        Also copies feature_norm_params from the DataModule and builds
        spatial index for nearest-neighbor trend estimation.

        Added mask verification to ensure HuggingFace models correctly
        use past_observed_mask to zero out padded positions in attention.
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

        # NOTE: TrendModel is now used for trend prediction in _compute_batch_trends()
        # The sophisticated logic (Mann-Kendall testing, optimal window selection, spatial interpolation)
        # is in TrendModel._predict_trend(), which we delegate to at inference time.
        # This means we don't need loc_trend_params, loc_coords, or _find_nearest_neighbor_trend anymore.

        # Verify that the model's forward pass accepts and uses past_observed_mask
        # by checking that outputs differ when mask zeros out all timesteps
        self._verify_mask_is_used()

    def _verify_mask_is_used(self):
        """Smoke test: masked-out inputs should produce different outputs than unmasked."""
        # Skip mask verification if model hasn't been fully built yet
        # (e.g., during checkpoint load before _build_model completes)
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
                f"on model output. Padded positions may be contributing spurious signal. "
                f"Verify HuggingFace implementation handles this mask correctly."
            )
        else:
            logging.info(f"[{self.config.model_type}] Mask verification passed.")

    def _compute_batch_trends(self, adm_ids, years: torch.Tensor, dm, lats, lons) -> torch.Tensor:
        """
        Compute normalised trend estimate for each sample in a batch.

        Uses TrendModel's sophisticated prediction logic which includes:
        - Mann-Kendall significance testing for trend validation
        - Optimal window selection for robust trend estimation
        - Forward/backward interpolation for test years between training blocks
        - Nearest-neighbor spatial interpolation for unseen locations

        Args:
            adm_ids: List of administrative region IDs
            years: Tensor of years (batch,)
            dm: DataModule with y_mean and y_std
            lats: Tensor of latitudes (batch,)
            lons: Tensor of longitudes (batch,)

        Returns:
            Tensor of shape (batch, 1) with trends in z-score space
        """
        # Guard against calling when trend is disabled
        if not self.config.use_residual_trend:
            raise RuntimeError(
                "_compute_batch_trends called with use_residual_trend=False. "
                "This is a bug in _shared_step - trend should not be computed or added."
            )

        # Construct test items for TrendModel
        test_items = []
        for i, (loc, year) in enumerate(zip(adm_ids, years)):
            year_int = int(year.item()) if hasattr(year, 'item') else int(year)
            test_items.append({
                KEY_LOC: loc,
                KEY_YEAR: year_int
            })

        # Use TrendModel's sophisticated prediction logic
        trend_predictions_orig = self.trend_model._predict_trend(test_items).flatten()

        # Normalize to z-score space
        trends_z = (trend_predictions_orig - dm.y_mean) / dm.y_std
        return torch.tensor(trends_z, dtype=torch.float32, device=self.device).unsqueeze(1)

    # -- Loss ----------------------------------------------------------------

    def _compute_weighted_loss(self, pred: torch.Tensor, y: torch.Tensor,
                               validity_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss between predictions and targets.

        Note: The validity_mask indicates which timesteps in the input sequence
        are real vs padded. However, since the model outputs a SCALAR prediction
        per sample (yield), weighting by the fraction of valid timesteps would
        arbitrarily down-weight short-season samples. A 191-day season is not
        inherently less trustworthy than a 365-day season.

        Using plain MSE is scientifically more defensible. The mask is retained
        for attention masking (in forward()) but not used for loss weighting.
        """
        return F.mse_loss(pred, y)

    # -- Lightning steps -----------------------------------------------------

    def _shared_step(self, batch, metrics: ModelMetrics, loss_key: str):
        x_ts, x_static, y, years, adm_ids, lats, lons, validity_mask = batch
        dm = self.trainer.datamodule

        # Compute trends only if use_residual_trend is enabled
        if self.config.use_residual_trend:
            batch_trends = self._compute_batch_trends(adm_ids, years, dm, lats, lons)
            assert batch_trends is not None, (
                "use_residual_trend=True but _compute_batch_trends returned None"
            )
        else:
            batch_trends = None

        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        # Pass observed_mask to forward so models can use it for attention masking
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        # Only add trend back when use_residual_trend is True
        # Detach batch_trends to prevent gradient flow through OLS-computed values
        if batch_trends is not None:
            final_pred = pred + batch_trends.squeeze(-1).detach()
        else:
            final_pred = pred
        loss = self._compute_weighted_loss(final_pred, y, validity_mask)

        metrics.update(final_pred.detach(), y.detach())
        self.log(loss_key, loss, prog_bar=True)
        return loss

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
            assert batch_trends is not None, (
                "use_residual_trend=True but _compute_batch_trends returned None"
            )
        else:
            batch_trends = None

        # Forward pass (same as training)
        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        # Add trend back
        final_pred_z = pred + batch_trends.squeeze(-1) if batch_trends is not None else pred

        # Compute loss in z-score space (for consistency with training)
        loss = self._compute_weighted_loss(final_pred_z, y_z, validity_mask)

        # Denormalize to original scale for metrics computation
        # Ensure y_std and y_mean are on the same device as the predictions
        device = final_pred_z.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)
        final_pred_orig = final_pred_z.detach() * y_std + y_mean
        y_orig = y_z.detach() * y_std + y_mean

        # Clip predictions to physically meaningful range (yields ≥ 0)
        final_pred_clipped = torch.clamp(final_pred_orig, min=0.0)

        # Log clip rate as diagnostic (helps identify model issues)
        # Only count predictions that were actually clipped from negative values
        # (not all zeros, since legitimate zero yields are possible)
        clipped_mask = final_pred_orig < 0.0
        clip_rate = clipped_mask.float().mean()
        self.log(f'{stage}/clip_rate', clip_rate, prog_bar=False)

        # Also log stats about negative predictions before clipping
        negative_rate = (final_pred_orig < 0).float().mean()
        self.log(f'{stage}/negative_rate', negative_rate, prog_bar=False)

        # Update metrics with clipped predictions (in original scale)
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
        # Training uses _shared_step without clipping (honest gradients)
        return self._shared_step(batch, self.train_metrics, 'train_loss')

    def on_train_epoch_end(self):
        results = self.train_metrics.compute()
        self.log('train/mse', results['mse'], prog_bar=False)
        self.log('train/mae', results['mae'], prog_bar=False)
        self.log('train/r2', results['r2'], prog_bar=False)
        self.log('train/rmse', torch.sqrt(results['mse']).item(), prog_bar=False)
        self.log('train/mape', results['mape'], prog_bar=False)
        self.log('train/smape', results['smape'], prog_bar=False)
        self.train_metrics.log_results(step="train")
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        # Evaluation uses clipping for physically meaningful predictions
        return self._eval_step_with_clipping(batch, self.val_metrics, 'val_loss', stage='val')

    def on_validation_epoch_end(self):
        results = self.val_metrics.compute()
        self.log('val/mse', results['mse'], prog_bar=False)
        self.log('val/mae', results['mae'], prog_bar=False)
        self.log('val/r2', results['r2'], prog_bar=False)
        self.log('val/rmse', torch.sqrt(results['mse']).item(), prog_bar=False)
        self.log('val/mape', results['mape'], prog_bar=False)
        self.log('val/smape', results['smape'], prog_bar=False)
        self.log('val/nrmse', results['nrmse'], prog_bar=False)
        self.val_metrics.log_results(step="val")
        self.val_metrics.reset()

    def _compute_per_year_metrics_from_preds(self) -> dict:
        """Compute per-year metrics from accumulated predictions using torchmetrics."""
        results = {}
        all_preds = []
        all_targets = []

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

            # SMAPE - manual computation for consistency
            smape_val = torch.mean(2.0 * torch.abs(preds - targets) /
                                  (torch.abs(preds) + torch.abs(targets) + 1e-6))
            nrmse_val = rmse_val / (torch.mean(targets) + 1e-6)

            # Store per-year metrics
            results[f'nrmse_{year}'] = nrmse_val.item()
            results[f'mape_{year}'] = mape_val.item()
            results[f'r2_{year}'] = r2_val.item()
            results[f'rmse_{year}'] = rmse_val.item()
            results[f'mae_{year}'] = mae_val.item()
            results[f'mse_{year}'] = mse_val.item()
            results[f'smape_{year}'] = smape_val.item()

            # Accumulate for overall metrics
            all_preds.extend(data['preds'])
            all_targets.extend(data['targets'])

        # Compute overall metrics
        if all_preds and all_targets:
            all_preds_t = torch.tensor(all_preds)
            all_targets_t = torch.tensor(all_targets)

            mse = MeanSquaredError()
            r2 = R2Score()
            mae = MeanAbsoluteError()
            mape = MeanAbsolutePercentageError()

            mse_val = mse(all_preds_t, all_targets_t)
            r2_val = r2(all_preds_t, all_targets_t)
            mae_val = mae(all_preds_t, all_targets_t)
            mape_val = mape(all_preds_t, all_targets_t)
            rmse_val = torch.sqrt(mse_val)

            smape_val = torch.mean(2.0 * torch.abs(all_preds_t - all_targets_t) /
                                  (torch.abs(all_preds_t) + torch.abs(all_targets_t) + 1e-6))
            nrmse_val = rmse_val / (torch.mean(all_targets_t) + 1e-6)

            results['nrmse_overall'] = nrmse_val.item()
            results['mape_overall'] = mape_val.item()
            results['r2_overall'] = r2_val.item()
            results['rmse_overall'] = rmse_val.item()
            results['mae_overall'] = mae_val.item()
            results['mse_overall'] = mse_val.item()
            results['smape_overall'] = smape_val.item()

        return results

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

    def on_test_epoch_end(self):
        results = self.test_metrics.compute()
        self.log('test/mse', results['mse'], prog_bar=False)
        self.log('test/mae', results['mae'], prog_bar=False)
        self.log('test/r2', results['r2'], prog_bar=False)
        self.log('test/rmse', torch.sqrt(results['mse']).item(), prog_bar=False)
        self.log('test/mape', results['mape'], prog_bar=False)
        self.log('test/smape', results['smape'], prog_bar=False)
        self.log('test/nrmse', results['nrmse'], prog_bar=False)
        self.test_metrics.log_results(step="test")
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
            assert batch_trends is not None, (
                "use_residual_trend=True but _compute_batch_trends returned None"
            )
        else:
            batch_trends = None

        # Forward pass (same as training)
        x_ts_n = self._normalize_time_series(x_ts, observed_mask=validity_mask)
        x_static_n = self._normalize_and_impute_static(x_static)
        pred = self.forward(x_ts_n, x_static_n, observed_mask=validity_mask)

        # Add trend back
        final_pred_z = pred + batch_trends.squeeze(-1) if batch_trends is not None else pred

        # Denormalize to original scale
        device = final_pred_z.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)
        predictions_orig = final_pred_z.detach() * y_std + y_mean
        targets_orig = y_z.detach() * y_std + y_mean

        # Clip predictions to physically meaningful range (yields >= 0)
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

    def on_test_start(self):
        """Initialize prediction cache for recursive lag prediction and per-year metrics."""
        # Initialize per-year prediction storage for CSV results
        dm = self.trainer.datamodule
        if hasattr(dm, '_test_years') and dm._test_years is not None:
            self._test_years = dm._test_years
            self._per_year_preds = {year: {'preds': [], 'targets': []} for year in self._test_years}
            logging.info(f"[Per-Year Metrics] Initialized storage for test years: {sorted(self._test_years)}")
        else:
            logging.warning("[Per-Year Metrics] Datamodule has no _test_years set, per-year metrics will not be computed")
            self._test_years = set()  # Initialize as empty set to prevent AttributeError in downstream methods
            self._per_year_preds = {}

        if self.config.use_recursive_lags:
            self._prediction_cache = {}
            logging.info(f"[Recursive Lags] Initialized prediction cache for true out-of-sample testing")
            if self._test_years:  # Empty set is falsy, so this safely checks if test years exist
                logging.info(f"[Recursive Lags] Test years: {sorted(self._test_years)}")
            else:
                logging.warning("[Recursive Lags] No test years available for recursive prediction")

    def _replace_lags_with_predictions(self, x_static: torch.Tensor, years: torch.Tensor,
                                       adm_ids: List[str]) -> torch.Tensor:
        """
        Replace lag features with cached predictions for recursive lag evaluation.

        For test samples where lag years fall within the test set, this replaces
        the observed lag values (which cause leakage) with previously predicted values.

        Args:
            x_static: Static features tensor (batch, n_static_features)
            years: Tensor of years (batch,)
            adm_ids: List of administrative region IDs

        Returns:
            Modified x_static with lag features replaced by predictions where appropriate
        """
        if not self.config.use_recursive_lags or self.config.lag_years == 0:
            return x_static

        x_static_modified = x_static.clone()
        static_feature_names = self._get_static_feature_names()

        # Find indices of lag features in the static feature array
        lag_feature_indices = []
        for lag in range(1, self.config.lag_years + 1):
            lag_name = f'lag_yield_{lag}'
            if lag_name in static_feature_names:
                lag_feature_indices.append((lag, static_feature_names.index(lag_name)))

        if not lag_feature_indices:
            return x_static

        # Replace each lag feature with cached prediction if available
        for i, (year, adm_id) in enumerate(zip(years, adm_ids)):
            year_int = int(year.item()) if hasattr(year, 'item') else int(year)

            for lag, lag_idx in lag_feature_indices:
                lag_year = year_int - lag

                # Check if this lag year is in the test set
                if self._test_years and lag_year in self._test_years:
                    cache_key = (adm_id, lag_year)

                    if cache_key in self._prediction_cache:
                        # Use cached prediction (already in z-score space)
                        x_static_modified[i, lag_idx] = self._prediction_cache[cache_key]
                    else:
                        # No prediction available yet - use default (mean in z-score space = 0.0)
                        # This can happen if batch isn't perfectly sorted chronologically within each location
                        x_static_modified[i, lag_idx] = 0.0

                        logging.warning(
                            f"[Recursive Lags] No cached prediction for {adm_id} year {lag_year}, "
                            f"using default (mean in z-score space=0.0). "
                            f"This may occur if test batches are not sorted chronologically within each location."
                        )

        return x_static_modified

    def _cache_predictions(self, predictions_z: torch.Tensor, years: torch.Tensor,
                          adm_ids: List[str], dm):
        """
        Cache predictions in ORIGINAL scale for recursive lag replacement.

        x_static holds raw (un-normalized) lag yields. _normalize_and_impute_static
        will z-score them later. Storing in original scale ensures only one
        normalization pass occurs.

        Args:
            predictions_z: Predictions in z-score space [B]
            years: Years [B]
            adm_ids: Location IDs [B]
            dm: DataModule for denormalization
        """
        if not self.config.use_recursive_lags or self.config.lag_years == 0:
            return

        device = predictions_z.device
        y_std = dm.y_std.to(device) if hasattr(dm.y_std, 'to') else float(dm.y_std)
        y_mean = dm.y_mean.to(device) if hasattr(dm.y_mean, 'to') else float(dm.y_mean)

        # Convert to original scale
        predictions_orig = predictions_z.detach() * y_std + y_mean

        for pred, year, adm_id in zip(predictions_orig, years, adm_ids):
            year_int = int(year.item()) if hasattr(year, 'item') else int(year)

            # Only cache predictions for test years
            if self._test_years and year_int in self._test_years:
                cache_key = (adm_id, year_int)
                self._prediction_cache[cache_key] = pred.detach().cpu().item()

                # Log first few cached predictions for debugging
                if len(self._prediction_cache) <= 5:
                    logging.debug(
                        f"[Recursive Lags] Cached prediction for {adm_id} year {year_int}: "
                        f"orig_scale={pred.detach().cpu().item():.4f}"
                    )

    def test_step(self, batch, batch_idx):
        """Test step with optional recursive lag prediction and per-year accumulation."""
        if not self.config.use_recursive_lags or self.config.lag_years == 0:
            # Use standard evaluation and accumulate per-year predictions
            loss, preds, targets, years = self._eval_step_with_clipping(
                batch, self.test_metrics, 'test_loss', stage='test', return_orig=True
            )
            # Accumulate per-year predictions for CSV results
            self._accumulate_per_year_predictions(preds, targets, years)
            return loss

        # Recursive lag mode: modify batch to use cached predictions
        x_ts, x_static, y_z, years, adm_ids, lats, lons, validity_mask = batch
        dm = self.trainer.datamodule

        # Replace lag features with cached predictions
        x_static_modified = self._replace_lags_with_predictions(x_static, years, adm_ids)

        # Create modified batch
        modified_batch = (x_ts, x_static_modified, y_z, years, adm_ids, lats, lons, validity_mask)

        # Run evaluation step and get both z-score and original predictions
        loss, preds_z, preds_clipped, targets, years = self._eval_step_with_clipping(
            modified_batch, self.test_metrics, 'test_loss', stage='test',
            return_orig=True, return_predictions=True
        )

        # Cache predictions in z-score space for recursive lag prediction
        # The cached predictions are used for subsequent test samples in the same location
        self._cache_predictions(preds_z, years, adm_ids, dm)

        # Accumulate per-year predictions for CSV results (using clipped predictions in original scale)
        self._accumulate_per_year_predictions(preds_clipped, targets, years)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr,
                                 weight_decay=self.weight_decay)

    def on_fit_start(self):
        """Log model size to wandb at the start of training."""
        print(f"\n{'=' * 60}")
        print("DEBUG: on_fit_start() called - counting model parameters...")
        print(f"{'=' * 60}")

        # Debug: Check what modules exist
        print(f"DEBUG: Model type: {type(self).__name__}")
        print(f"DEBUG: Has base_model: {hasattr(self, 'base_model')}")
        print(f"DEBUG: Has regression_head: {hasattr(self, 'regression_head')}")

        if hasattr(self, 'base_model'):
            print(f"DEBUG: base_model type: {type(self.base_model).__name__}")
            base_model_params = sum(p.numel() for p in self.base_model.parameters())
            print(f"DEBUG: base_model parameters: {base_model_params:,}")

        if hasattr(self, 'regression_head'):
            regression_head_params = sum(p.numel() for p in self.regression_head.parameters())
            print(f"DEBUG: regression_head parameters: {regression_head_params:,}")

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
            print("DEBUG: Logged to wandb successfully")


# =========================================================
# MODEL ARCHITECTURES
# =========================================================

class AutoformerYieldModel(BaseTimeSeriesModel):
    """Autoformer: auto-correlation based transformer for yield forecasting."""

    def _build_model(self) -> nn.Module:
        # First, load or create the base model
        if self.config.load_checkpoint:
            model = HFAutoformerModel.from_pretrained(self.config.load_checkpoint)
            # Extract config values from loaded model
            self._actual_lags = list(model.config.lags_sequence)
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)
        else:
            # Use HFAutoformerModel, NOT AutoformerForPrediction
            # AutoformerForPrediction wraps outputs in distribution interface and detaches gradients
            # HFAutoformerModel returns raw encoder/decoder outputs with gradients preserved

            # Following baseline approach: process temporal features first,
            # then concatenate with static features AFTER getting pooled representation
            cfg = AutoformerConfig(
                prediction_length=1,
                context_length=self.config.seq_len,
                lags_sequence=[1],
                input_size=self.n_ts_features,  # Only temporal features
                num_time_features=0,
                num_static_categorical_features=0,
                # NOT using num_static_real_features - we'll concatenate later
                d_model=64, num_attention_heads=4, ffn_dim=256, num_layers=3,
                dropout=0.1,
            )

            # Subtract 1 from context_length to accommodate lag=1
            # HF constraint: context_length + max(lags_sequence) <= seq_len
            # With seq_len=365, lag=1: context_length must be <= 364
            # This matches the working script's approach (line 624 in cybenchAutoformer.py)
            cfg.context_length = int(getattr(cfg, "context_length", self.config.seq_len)) - 1

            model = HFAutoformerModel(cfg)

            # READ BACK what HF actually stored (it may override our values)
            self._actual_lags = list(model.config.lags_sequence)
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)

            logging.info(f"[Autoformer BUILD] CONFIG: seq_len={self.config.seq_len}, "
                        f"context_length={self._actual_context_length}, lags={self._actual_lags}, "
                        f"n_ts_features={self.n_ts_features}, n_static_features={self.n_static_features}")

        # Always create regression head, even when loading from checkpoint
        # This ensures parameters are registered before optimizer is created
        # d_model is hardcoded to 64 in the config above
        d_model = 64
        combined_dim = d_model + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, 1)
        )

        logging.info(f"[Autoformer BUILD] Created regression head: d_model={d_model}, "
                    f"combined_dim={combined_dim}, hidden_dim={combined_dim // 2}")

        self._model_ready = True
        return model

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        Following baseline pattern: temporal → pool → concat static → regression

        Args:
            x_ts: Time series features, shape (batch, seq_len, n_ts_features)
            x_static: Static features, shape (batch, n_static_features)
            observed_mask: Boolean mask of shape (batch, seq_len) for valid timesteps

        Returns:
            Predictions of shape (batch,)
        """
        batch_size, seq_len = x_ts.shape[:2]

        # Step 1: Process temporal features through Autoformer (NO static features yet)
        # Request encoder outputs explicitly
        outputs = self.base_model(
            past_values=x_ts,
            past_time_features=torch.zeros(batch_size, seq_len, 0, device=x_ts.device),
            past_observed_mask=observed_mask.unsqueeze(-1).expand(-1, -1, x_ts.shape[2]).float() if observed_mask is not None else None,
            future_values=torch.zeros(batch_size, 1, x_ts.shape[-1], device=x_ts.device),
            future_time_features=torch.zeros(batch_size, 1, 0, device=x_ts.device),
            return_dict=True,
            output_hidden_states=True,  # Request all hidden states
        )

        # Step 2: Extract hidden state using shared helper
        h = self._extract_hidden_state(outputs)

        # Step 3: Pool hidden state using shared helper
        pooled = self._pool_hidden_state(h)  # (B, d_model)

        # Step 4: Concatenate with static features and pass through regression head
        combined = torch.cat([pooled, x_static], dim=-1)
        return self.regression_head(combined).squeeze(-1)


class PatchTSTModel(BaseTimeSeriesModel):
    """PatchTST: patch-based transformer (linear complexity in sequence length)."""

    def _build_model(self) -> nn.Module:
        # First, load or create the base model
        if self.config.load_checkpoint:
            model = HFPatchTSTModel.from_pretrained(self.config.load_checkpoint)
        else:
            patch_len = {"daily": 16, "weekly": 4, "dekad": 6}[self.config.aggregation]
            stride = {"daily": 8, "weekly": 2, "dekad": 3}[self.config.aggregation]

            logging.info(f"[PatchTST BUILD] CONFIG: seq_len={self.config.seq_len}, "
                        f"n_ts_features={self.n_ts_features}, n_static_features={self.n_static_features}, "
                        f"patch_length={patch_len}, stride={stride}, aggregation={self.config.aggregation}")

            cfg = PatchTSTConfig(
                prediction_length=1,
                context_length=self.config.seq_len,
                # num_input_channels is the total number of input channels
                # This should match n_ts_features (excluding time features since we pass them separately)
                num_input_channels=self.n_ts_features,
                num_time_features=self.num_time_features,
                # NOT using num_static_real_features - we'll concatenate later
                d_model=64, num_attention_heads=4, ffn_dim=256, num_layers=3,
                dropout=0.1,
                patch_length=patch_len,
                stride=stride,
            )

            model = HFPatchTSTModel(cfg)

            logging.info(f"[PatchTST] Config verification: context_length={model.config.context_length}, "
                        f"num_input_channels={model.config.num_input_channels}, num_time_features={model.config.num_time_features}")

        # Probe the actual output shape to build regression head correctly
        # PatchTST can output different shapes depending on configuration:
        #   - (B, seq_len, d_model) for some configs
        #   - (B, n_channels, n_patches, d_model) for multivariate patch-based
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.seq_len, self.n_ts_features)
            out = model(past_values=dummy)
            h = self._extract_hidden_state(out)
            pooled = self._pool_hidden_state(h)
            pooled_dim = pooled.shape[-1]

        logging.info(f"[PatchTST BUILD] Probed actual output: h.shape={h.shape} → pooled.shape={pooled.shape} → pooled_dim={pooled_dim}")

        # Build regression head with correct dimension (always created, even when loading from checkpoint)
        combined_dim = pooled_dim + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, 1)
        )

        # Store for validation in forward()
        self._pooled_dim = pooled_dim

        logging.info(f"[PatchTST BUILD] Created regression head: pooled_dim={pooled_dim}, "
                    f"combined_dim={combined_dim}, hidden_dim={combined_dim // 2}")

        self._model_ready = True
        return model

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through PatchTST model.
        Following baseline pattern: temporal → pool → concat static → regression
        """
        # Step 1: Process temporal features through PatchTST
        # Pass past_observed_mask to prevent padded zeros from being used in attention
        past_observed_mask = observed_mask.unsqueeze(-1).expand(-1, -1, x_ts.shape[2]).float() if observed_mask is not None else None
        outputs = self.base_model(past_values=x_ts, past_observed_mask=past_observed_mask, future_values=None)

        # Step 2: Extract hidden state using shared helper
        h = self._extract_hidden_state(outputs)

        # Step 3: Pool hidden state using shared helper
        pooled = self._pool_hidden_state(h)  # (B, pooled_dim)

        # Validate against probed dimension (catches shape regressions)
        if pooled.shape[-1] != self._pooled_dim:
            raise RuntimeError(
                f"[PatchTST] Pooled dimension mismatch! "
                f"Expected {self._pooled_dim} (from build-time probe), "
                f"got {pooled.shape[-1]}. This indicates the model output shape changed "
                f"between _build_model() and forward(). h.shape={h.shape}"
            )

        # Step 4: Concatenate with static features and pass through regression head
        combined = torch.cat([pooled, x_static], dim=-1)
        return self.regression_head(combined).squeeze(-1)



class TSMixerModel(BaseTimeSeriesModel):
    """TSMixer: all-MLP architecture — simple, fast, surprisingly competitive."""

    def _build_model(self) -> nn.Module:
        # First, load or create the base model
        if self.config.load_checkpoint:
            if TimeSeriesMixerModel is not None:
                model = TimeSeriesMixerModel.from_pretrained(self.config.load_checkpoint)
            else:
                model = TimeSeriesMixerForPrediction.from_pretrained(self.config.load_checkpoint)
        else:
            logging.info(f"[TSMixer BUILD] CONFIG: seq_len={self.config.seq_len}, "
                        f"n_ts_features={self.n_ts_features}, n_static_features={self.n_static_features}")

            # Use base model if available for gradient flow
            if TimeSeriesMixerModel is not None:
                model = TimeSeriesMixerModel(TimeSeriesMixerConfig(
                    prediction_length=1,
                    context_length=self.config.seq_len,
                    input_size=self.n_ts_features,
                    num_time_features=self.num_time_features,
                    # NOT using num_static_real_features - we'll concatenate later
                    hidden_size=64, num_layers=3, dropout=0.1, expansion_factor=2,
                ))
            else:
                logging.warning("TimeSeriesMixerModel not available, using ForPrediction")
                model = TimeSeriesMixerForPrediction(TimeSeriesMixerConfig(
                    prediction_length=1,
                    context_length=self.config.seq_len,
                    input_size=self.n_ts_features,
                    num_time_features=self.num_time_features,
                    num_static_categorical_features=0,
                    num_static_real_features=self.n_static_features,
                    hidden_size=64, num_layers=3, dropout=0.1, expansion_factor=2,
                ))

        # Probe the actual output shape to build regression head correctly
        # TSMixer can output different shapes depending on configuration and variant
        if TimeSeriesMixerModel is not None:
            # Base model - we can probe the output shape
            with torch.no_grad():
                dummy = torch.zeros(1, self.config.seq_len, self.n_ts_features)
                out = model(past_values=dummy)
                h = self._extract_hidden_state(out)
                pooled = self._pool_hidden_state(h)
                pooled_dim = pooled.shape[-1]

            logging.info(f"[TSMixer BUILD] Probed actual output: h.shape={h.shape} → pooled.shape={pooled.shape} → pooled_dim={pooled_dim}")

            # Build regression head with correct dimension
            combined_dim = pooled_dim + self.n_static_features
            self.regression_head = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.LayerNorm(combined_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(combined_dim // 2, 1)
            )

            # Store for validation in forward()
            self._pooled_dim = pooled_dim
            self._uses_for_prediction = False

            logging.info(f"[TSMixer BUILD] Created regression head: pooled_dim={pooled_dim}, "
                        f"combined_dim={combined_dim}, hidden_dim={combined_dim // 2}")
        else:
            # ForPrediction variant - outputs loc directly, no hidden state
            # We can't probe in the same way, so we use the known hidden_size
            hidden_size = 64
            combined_dim = hidden_size + self.n_static_features
            self.regression_head = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.LayerNorm(combined_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(combined_dim // 2, 1)
            )

            # Store for validation in forward()
            self._pooled_dim = None  # Not used for ForPrediction
            self._uses_for_prediction = True

            logging.info(f"[TSMixer BUILD] ForPrediction variant: using hidden_size={hidden_size}, "
                        f"combined_dim={combined_dim}, hidden_dim={combined_dim // 2}")

        self._model_ready = True
        return model

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x_ts: Time series features, shape (batch, seq_len, n_ts_features)
            x_static: Static features, shape (batch, n_static_features)
            observed_mask: Boolean mask of shape (batch, seq_len) for valid timesteps

        Returns:
            Predictions of shape (batch,)
        """
        # Step 1: Process temporal features through TSMixer
        outputs = self.base_model(past_values=x_ts)

        # Handle ForPrediction variant (no gradient flow through base model)
        if self._uses_for_prediction:
            logging.warning("Using ForPrediction variant - gradients will not flow through TSMixer")
            return outputs.loc.reshape(-1)

        # Step 2: Extract hidden state using shared helper
        h = self._extract_hidden_state(outputs)

        # Step 3: Pool hidden state using shared helper
        pooled = self._pool_hidden_state(h)  # (B, pooled_dim)

        # Validate against probed dimension (catches shape regressions)
        if pooled.shape[-1] != self._pooled_dim:
            raise RuntimeError(
                f"[TSMixer] Pooled dimension mismatch! "
                f"Expected {self._pooled_dim} (from build-time probe), "
                f"got {pooled.shape[-1]}. This indicates the model output shape changed "
                f"between _build_model() and forward(). h.shape={h.shape}"
            )

        # Step 4: Concatenate with static features and pass through regression head
        combined = torch.cat([pooled, x_static], dim=-1)
        return self.regression_head(combined).squeeze(-1)

class InformerModel(BaseTimeSeriesModel):
    """Informer: sparse attention transformer, efficient on long sequences."""

    def _build_model(self) -> nn.Module:
        # First, load or create the base model
        if self.config.load_checkpoint:
            model = HFInformerModel.from_pretrained(self.config.load_checkpoint)
            # Extract config values from loaded model
            self._actual_lags = list(model.config.lags_sequence)
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)
        else:
            # context_length must leave room for lags
            # HF constraint: context_length + max(lags_sequence) <= seq_len
            # So: context_length = seq_len - max(lags_sequence)
            requested_lags = [1]
            context_length = self.config.seq_len - max(requested_lags)

            # Build model with adjusted context_length (BASELINE PATTERN)
            cfg = InformerConfig(
                prediction_length=1,
                context_length=context_length,
                lags_sequence=requested_lags,
                input_size=self.n_ts_features,  # Only temporal features
                num_time_features=0,
                # NOT using num_static_real_features - we'll concatenate later
                d_model=64, num_attention_heads=4, ffn_dim=256, num_layers=3,
                dropout=0.1,
            )
            # Ensure context_length accommodates lags
            # Already computed: context_length = seq_len - max(lags_sequence)
            # This should be correct, but verify after model creation
            model = HFInformerModel(cfg)  # Use base model for gradient flow

            # READ BACK what HF actually stored (it may override our values)
            self._actual_lags = list(model.config.lags_sequence)
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)

            logging.info(f"[Informer BUILD] CONFIG: seq_len={self.config.seq_len}, "
                        f"context_length={self._actual_context_length}, lags={self._actual_lags}, "
                        f"num_time_features={self._actual_num_time_features}, n_ts_features={self.n_ts_features}")

        # Always create regression head, even when loading from checkpoint
        d_model = 64  # from config above
        combined_dim = d_model + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, 1)
        )

        logging.info(f"[Informer BUILD] Created regression head: d_model={d_model}, "
                    f"combined_dim={combined_dim}, hidden_dim={combined_dim // 2}")

        self._model_ready = True
        return model

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through Informer model. Following baseline pattern."""
        batch_size, seq_len = x_ts.shape[:2]

        # Step 1: Process temporal features through Informer
        past_time_features = torch.zeros(batch_size, seq_len, self._actual_num_time_features, device=x_ts.device, dtype=x_ts.dtype)
        past_observed_mask = observed_mask.unsqueeze(-1).expand(-1, -1, x_ts.shape[2]).float() if observed_mask is not None else None

        outputs = self.base_model(past_values=x_ts, past_time_features=past_time_features, past_observed_mask=past_observed_mask, return_dict=True)

        # Step 2: Extract hidden state using shared helper
        h = self._extract_hidden_state(outputs)

        # Step 3: Pool hidden state using shared helper
        pooled = self._pool_hidden_state(h)  # (B, d_model)

        # Step 4: Concatenate with static features and pass through regression head
        combined = torch.cat([pooled, x_static], dim=-1)
        return self.regression_head(combined).squeeze(-1)


class TSTModel(BaseTimeSeriesModel):
    """
    TimeSeriesTransformer: vanilla encoder-decoder with student-t output.

    Operates on raw normalised float values — no tokenisation.
    distribution_output='student_t' is robust to yield outliers.
    We extract the mean (index 0) of the distribution parameters for a
    deterministic prediction.
    """

    def _build_model(self) -> nn.Module:
        # First, load or create the base model
        if self.config.load_checkpoint:
            model = HFTimeSeriesTransformerModel.from_pretrained(self.config.load_checkpoint)
            # Extract config values from loaded model
            self._actual_lags = list(model.config.lags_sequence)
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)
        else:
            # context_length must leave room for lags
            # HF constraint: context_length + max(lags_sequence) <= seq_len
            # So: context_length = seq_len - max(lags_sequence)
            requested_lags = [1]
            context_length = self.config.seq_len - max(requested_lags)

            # Build model with adjusted context_length (BASELINE PATTERN)
            cfg = TimeSeriesTransformerConfig(
                prediction_length=1,
                context_length=context_length,
                lags_sequence=requested_lags,
                input_size=self.n_ts_features,  # Only temporal features
                num_time_features=0,
                # NOT using num_static_real_features - we'll concatenate later
                d_model=64, num_attention_heads=4, num_hidden_layers=3,
                dim_feedforward=256, dropout=0.1, attention_probs_dropout_prob=0.1,
                activation_function="gelu", layer_norm_eps=1e-5,
                scaling="std", loss="nll", distribution_output="student_t",
            )
            # Ensure context_length accommodates lags
            # Already computed: context_length = seq_len - max(lags_sequence)
            # This should be correct, but verify after model creation
            model = HFTimeSeriesTransformerModel(cfg)  # Use base model for gradient flow

            # READ BACK what HF actually stored (it may override our values)
            self._actual_lags = list(model.config.lags_sequence)
            self._actual_num_time_features = int(model.config.num_time_features)
            self._actual_context_length = int(model.config.context_length)

            logging.info(f"[TST BUILD] CONFIG: seq_len={self.config.seq_len}, "
                        f"context_length={self._actual_context_length}, lags={self._actual_lags}, "
                        f"num_time_features={self._actual_num_time_features}, n_ts_features={self.n_ts_features}")

        # Always create regression head, even when loading from checkpoint
        d_model = 64  # from config above
        combined_dim = d_model + self.n_static_features
        self.regression_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.LayerNorm(combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, 1)
        )

        logging.info(f"[TST BUILD] Created regression head: d_model={d_model}, "
                    f"combined_dim={combined_dim}, hidden_dim={combined_dim // 2}")

        self._model_ready = True
        return model

    def forward(self, x_ts: torch.Tensor, x_static: torch.Tensor,
                observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through TST model. Following baseline pattern."""
        batch_size, seq_len = x_ts.shape[:2]

        # Step 1: Process temporal features through TST
        past_time_features = torch.zeros(batch_size, seq_len, self._actual_num_time_features, device=x_ts.device, dtype=x_ts.dtype)
        past_observed_mask = observed_mask.unsqueeze(-1).expand(-1, -1, x_ts.shape[2]).float() if observed_mask is not None else None

        outputs = self.base_model(past_values=x_ts, past_time_features=past_time_features, past_observed_mask=past_observed_mask, return_dict=True)

        # Step 2: Extract hidden state using shared helper
        h = self._extract_hidden_state(outputs)

        # Step 3: Pool hidden state using shared helper
        pooled = self._pool_hidden_state(h)  # (B, d_model)

        # Step 4: Concatenate with static features and pass through regression head
        combined = torch.cat([pooled, x_static], dim=-1)
        return self.regression_head(combined).squeeze(-1)

def create_model(config: TSTModelConfig) -> BaseTimeSeriesModel:
    model_map = {
        "autoformer": AutoformerYieldModel,
        "patchtst": PatchTSTModel,
        "informer": InformerModel,
        "tst": TSTModel,
    }
    # Only register TSMixer if import succeeded
    if TimeSeriesMixerForPrediction is not None:
        model_map["tsmixer"] = TSMixerModel

    if config.model_type.lower() not in model_map:
        raise ValueError(f"Unknown model_type '{config.model_type}'. "
                         f"Choose from: {list(model_map)}")
    return model_map[config.model_type.lower()](
        config, lr=config.lr, weight_decay=config.weight_decay)
