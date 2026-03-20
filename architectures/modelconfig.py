from typing import Optional, Dict, List, Tuple

from dataclasses import dataclass

from cybench.config import (
    GDD_BASE_TEMP, GDD_UPPER_LIMIT, LOCATION_PROPERTIES, SOIL_PROPERTIES,
    FORECAST_LEAD_TIME, KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, KEY_CROP_SEASON,
    CROP_CALENDAR_DATES
)

# %% Global constants
# Weather feature lists - used as defaults by ModelConfig.weather_features property
# These are module-level constants; actual features used come from config
WEATHER_FEATURES_BASE = ['tmin', 'tmax', 'tavg', 'prec', 'rad']
WEATHER_FEATURES_WITH_CWB = ['tmin', 'tmax', 'tavg', 'prec', 'cwb', 'rad']

# Remote sensing features - always included
REMOTE_SENSING_FEATURES = ['fpar', 'ndvi', 'ssm', 'rsm']

STANDARD_STATIC_VARS = SOIL_PROPERTIES + LOCATION_PROPERTIES
# CROP_CALENDAR_DATES imported from cybench.config

# based on config.weather_features and config.time_series_vars properties
print(f"[Feature Config] Static vars ({len(STANDARD_STATIC_VARS)}): {STANDARD_STATIC_VARS}")

@dataclass
class TSTModelConfig:
    """Central configuration for time series forecasting model."""
    crop: str = "maize"
    country: str = "NL"
    model_type: str = "autoformer"
    aggregation: str = "dekad"
    use_sota_features: bool = False
    include_spatial_features: bool = False
    use_residual_trend: bool = True
    lag_years: int = 1  # Default to 1, max 2
    load_checkpoint: Optional[str] = None
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 0  # Default 0 for HPC compatibility
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 50
    test_years: int = 3
    # Feature toggles for ablation studies
    use_cwb_feature: bool = False  # Include climate water balance
    drop_tavg: bool = False  # Drop tavg if dataset computes it as (tmin+tmax)/2
    # Recursive lag prediction for true out-of-sample testing
    use_recursive_lags: bool = False  # Use predicted yields as lags during testing (default: False for backward compat)
    # Domain feature engineering flags
    use_gdd: bool = False # GDD time series channel
    use_heat_stress_days: bool = False  # Heat stress static counts
    use_rue: bool = False           # RUE index time series channel
    use_farquhar: bool = False      # Farquhar proxy time series channel 
    # Results directory for CSV output
    results_dir: str = "checkpoints/results"

    @property
    def seq_len(self):
        """Sequence length derived from aggregation frequency."""
        return {"daily": 365, "weekly": 52, "dekad": 36}.get(self.aggregation, 365)

    @property
    def weather_features(self) -> List[str]:
        """
        Compute the list of weather features based on config flags.
        This is a computed property instead of a global variable,
        preventing timing issues where model init captures the wrong value.
        """
        features = list(WEATHER_FEATURES_WITH_CWB if self.use_cwb_feature
                       else WEATHER_FEATURES_BASE)
        if self.drop_tavg:
            features = [f for f in features if f != 'tavg']
        return features

    @property
    def time_series_vars(self) -> List[str]:
        """Full list of time series variables including remote sensing."""
        return self.weather_features + REMOTE_SENSING_FEATURES

    def _compute_expected_static_features(self) -> int:
        """
        Compute the total expected static feature count from the current config.
        This is used in setup() to validate that build_daily_input_sequence()
        actually produced the right number of features. Any mismatch means a
        feature creation step failed silently.
        """
        n_soil = len(SOIL_PROPERTIES)
        n_location = len(LOCATION_PROPERTIES)
        # Crop calendar: sos_date and eos_date use cyclic encoding (2 each),
        # other dates use 1 feature each
        n_crop = 0
        for date_name in CROP_CALENDAR_DATES:
            if date_name in ["sos_date", "eos_date"]:
                n_crop += 2  # sin and cos
            else:
                n_crop += 1
        n_spatial = 2 if self.include_spatial_features else 0
        n_lagged = self.lag_years
        # Heat stress: 7 scalar features when enabled
        n_heat_stress = 7 if self.use_heat_stress_days else 0 

        return n_soil + n_location + n_crop + n_spatial + n_lagged + n_heat_stress
    

@dataclass
class LinearModelConfig:
    """Central configuration for time series forecasting model."""
    crop: str = "maize"
    country: str = "NL"
    model_type: str = "nlinear"
    aggregation: str = "dekad"
    use_sota_features: bool = False
    include_spatial_features: bool = False
    use_residual_trend: bool = True
    lag_years: int = 1  # Default to 1, max 2 (constrained in CLI)
    load_checkpoint: Optional[str] = None
    seed: int = 42
    batch_size: int = 16
    num_workers: int = 0  # Default 0 for HPC compatibility
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 50
    test_years: int = 3
    # Feature toggles for ablation studies
    use_cwb_feature: bool = False  # Include crop water balance (redundant with prec+temp)
    drop_tavg: bool = False  # Drop tavg if dataset computes it as (tmin+tmax)/2
    use_revIN: bool = False  # Use RevIN normalization for XLinear endogenous series
    # Recursive lag prediction for true out-of-sample testing
    use_recursive_lags: bool = False  # Use predicted yields as lags during testing (default: False for backward compat)
    # Domain feature engineering flags
    use_gdd: bool = False # GDD time series channel
    use_heat_stress_days: bool = False  # Heat stress static counts
    use_rue: bool = False # RUE index time series channel
    use_farquhar: bool = False # Farquhar proxy time series channel 
    # Results directory for CSV output
    results_dir: str = "checkpoints/results"

    @property
    def seq_len(self):
        """Sequence length derived from aggregation frequency."""
        return {"daily": 365, "weekly": 52, "dekad": 36}.get(self.aggregation, 365)

    @property
    def weather_features(self) -> List[str]:
        """
        Compute the list of weather features based on config flags.
        """
        features = list(WEATHER_FEATURES_WITH_CWB if self.use_cwb_feature
                       else WEATHER_FEATURES_BASE)
        if self.drop_tavg:
            features = [f for f in features if f != 'tavg']
        return features

    @property
    def time_series_vars(self) -> List[str]:
        """Full list of time series variables including remote sensing."""
        return self.weather_features + REMOTE_SENSING_FEATURES

    def _compute_expected_static_features(self) -> int:
        """
        Compute the total expected static feature count from the current config.
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

        # Heat stress: 7 scalar features when enabled
        n_heat_stress = 7 if self.use_heat_stress_days else 0 

        return n_soil + n_location + n_crop + n_spatial + n_lagged + n_heat_stress
