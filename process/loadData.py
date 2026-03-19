import sys
import logging
from typing import Union

import numpy as np
import pandas as pd

from typing import Optional, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset
from cybench.config import (LOCATION_PROPERTIES, SOIL_PROPERTIES, CROP_CALENDAR_DATES)
from cybench.config import (
    GDD_BASE_TEMP, GDD_UPPER_LIMIT, LOCATION_PROPERTIES, SOIL_PROPERTIES,
    FORECAST_LEAD_TIME, KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, KEY_CROP_SEASON,
    CROP_CALENDAR_DATES
)

# Custom functions
sys.path.append('../architectures/')
from modelconfig import TSTModelConfig, LinearModelConfig

# %% Global constants
DEKAD_FREQ = "10D"
WEEKLY_FREQ = "W-MON"
DAILY_FREQ = "D"

# Maximum sequence lengths for padding — ensures uniform tensor shapes
MAX_SEQ_LENS = {"daily": 365, "weekly": 52, "dekad": 36}

# Weather feature lists - used as defaults by TSTModelConfig or LinearModelConfig.weather_features property
# These are module-level constants; actual features used come from config
WEATHER_FEATURES_BASE = ['tmin', 'tmax', 'tavg', 'prec', 'rad']
WEATHER_FEATURES_WITH_CWB = ['tmin', 'tmax', 'tavg', 'prec', 'cwb', 'rad']

# Remote sensing features - always included
REMOTE_SENSING_FEATURES = ['fpar', 'ndvi', 'ssm', 'rsm']

STANDARD_STATIC_VARS = SOIL_PROPERTIES + LOCATION_PROPERTIES
# CROP_CALENDAR_DATES imported from cybench.config

SOTA_TEMPORAL_VARS_LIST = [
    'sin_doy', 'cos_doy',
    'sin_month', 'cos_month',
    'season_sin', 'season_cos'
]

# based on config.weather_features and config.time_series_vars properties
print(f"[Feature Config] Static vars ({len(STANDARD_STATIC_VARS)}): {STANDARD_STATIC_VARS}")
print(f"[Feature Config] SOTA Temporal vars ({len(SOTA_TEMPORAL_VARS_LIST)}): {SOTA_TEMPORAL_VARS_LIST}")

# Sentinel value thresholds for remote sensing data
RS_SENTINEL_THRESHOLDS = {
    'fpar': (-0.1, 1.05),   # Physical bounds with small tolerance
    'ndvi': (-0.5, 1.05),   # Flag anything below -0.5 as sentinel
    'ssm':  (-0.1, 1.05),
    'rsm':  (-0.1, 1.05),
}
RS_VALID_RANGES = {
    'fpar': (0.0, 1.0),
    'ndvi': (0.1, 1.0),    # 0.1 minimum for vegetated agricultural surfaces
    'ssm':  (0.0, 1.0),
    'rsm':  (0.0, 1.0),
}

def prepare_features_and_targets(dataset):
    """
    Prepared features and target from the raw data.
    """
    X_list, y_list, years_list = [], [], []

    targets_array = dataset.targets()
    indices_list = list(dataset.indices())  # [(adm_id, year), ...]

    for i, idx in enumerate(indices_list):
        adm_id, year = idx
        target = targets_array[i]

        features = {}

        # Soil
        soil_row = dataset._dfs_x['soil'].loc[adm_id]
        for col in soil_row.index:
            features[f'soil_{col}'] = soil_row[col]

        # Meteorological
        meteo_rows = dataset._dfs_x['meteo'].loc[adm_id].loc[year]
        features['meteo_tmin_mean'] = meteo_rows['tmin'].mean()
        features['meteo_tmax_mean'] = meteo_rows['tmax'].mean()
        features['meteo_tavg_mean'] = meteo_rows['tavg'].mean()
        features['meteo_prec_sum'] = meteo_rows['prec'].sum()
        features['meteo_cwb_sum'] = meteo_rows['cwb'].sum()
        features['meteo_rad_sum'] = meteo_rows['rad'].sum()

        # Remote sensing
        for key in ['fpar', 'ndvi', 'ssm']:
            try:
                rs_rows = dataset._dfs_x[key].loc[adm_id].loc[year]
                features[f'{key}_mean'] = rs_rows.iloc[:, 0].mean() if not rs_rows.empty else np.nan
            except KeyError:
                features[f'{key}_mean'] = np.nan

        # Crop season
        try:
            cs_row = dataset._dfs_x['crop_season'].loc[(adm_id, year)]
            for col in cs_row.index:
                value = cs_row[col]
                if isinstance(value, pd.Timestamp):
                    value = (value - pd.Timestamp("1970-01-01")).days
                elif pd.isnull(value):
                    value = np.nan
                features[f'crop_{col}'] = value
        except KeyError:
            for col in dataset._dfs_x['crop_season'].columns:
                features[f'crop_{col}'] = np.nan

        X_list.append(list(features.values()))
        y_list.append(target)
        years_list.append(year)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y, years_list

# %% Shared parameters for time-series models

def _get_static_feature_names(
    include_spatial_features: bool,
    lag_years: int,
) -> List[str]:
    """
    Return static feature names in the EXACT order that _extract_static_features()
    appends values. This single source of truth is used by:
      - DailyCYBenchSeqDataModule._compute_feature_normalization()
      - DailyCYBenchSeqDataModule._get_static_feature_names() (thin wrapper)
      - BaseTimeSeriesModel._normalize_and_impute_static()
      - BaseTimeSeriesModel._get_static_feature_names() (thin wrapper)

    Keeping one implementation prevents the two classes from drifting out of sync,
    which would silently apply wrong normalization statistics to the wrong feature
    column.

    Feature order (must match _extract_static_features exactly):
      1. Soil properties
      2. Location properties
      3. Crop calendar dates (with cyclic encoding for sos_date and eos_date)
      4. Explicit lat/lon (conditional)
      5. Lagged yields (conditional)
    """
    names = list(SOIL_PROPERTIES)
    names.extend(LOCATION_PROPERTIES)
    # Crop calendar with cyclic encoding: sos_date and eos_date get sin/cos pairs
    for date_name in CROP_CALENDAR_DATES:
        if date_name in ["sos_date", "eos_date"]:
            names.extend([f'{date_name}_sin', f'{date_name}_cos'])
        else:
            names.append(date_name)

    if include_spatial_features:
        names.extend(['latitude_explicit', 'longitude_explicit'])

    for lag in range(1, lag_years + 1):
        names.append(f'lag_yield_{lag}')

    return names

def _clean_rs_series(series: pd.Series, var_name: str) -> pd.Series:
    """
    Mask sentinel values, then fill gaps, then clip to valid range.

    Previous implementation did ffill().bfill() BEFORE clipping, which
    could propagate sentinel values (e.g., -9999) across the season before
    being clipped. Now we mask first, then fill, then clip.

    Args:
        series: Raw remote sensing series
        var_name: Variable name for looking up thresholds

    Returns:
        Cleaned series with sentinels removed, gaps filled, and values clipped
    """
    s = series.copy().astype(float)
    lo, hi = RS_SENTINEL_THRESHOLDS.get(var_name, (-1e6, 1e6))

    # Step 1: mask out-of-physical-range values BEFORE any filling
    s[(s < lo) | (s > hi)] = np.nan

    # Step 2: fill gaps (now only real gaps, not sentinels)
    s = s.interpolate(method='linear', limit_direction='both')
    s = s.ffill().bfill()

    # Step 3: clip to valid agronomic range
    valid_lo, valid_hi = RS_VALID_RANGES.get(var_name, (lo, hi))
    s = s.clip(valid_lo, valid_hi)

    return s


def interpolate_to_daily(data: pd.Series, target_dates: pd.DatetimeIndex,
                         method: str = 'linear', interpolate_data: str = 'unknown') -> pd.Series:
    """
    Interpolate non-daily time series data to daily frequency.

    For remote sensing data, uses _clean_rs_series to properly handle
    sentinel values before filling.

    Args:
        data: Input series with non-daily frequency
        target_dates: DatetimeIndex for output
        method: Interpolation method ('linear' or 'ffill')
        interpolate_data: Data type for special handling ('fpar', 'ndvi', 'soil_moisture')

    Returns:
        Interpolated daily series
    """
    if isinstance(data.index, pd.MultiIndex):
        data = data.copy()
        data.index = pd.to_datetime(data.index.get_level_values(-1))
    else:
        data = data.copy()
        data.index = pd.to_datetime(data.index)

    data_daily = data.reindex(target_dates, method=None)

    # Use proper sentinel-aware cleaning for RS variables
    if interpolate_data in RS_SENTINEL_THRESHOLDS:
        data_daily = _clean_rs_series(data_daily, interpolate_data)
    elif interpolate_data == 'soil_moisture':
        data_daily = data_daily.interpolate(method='linear', limit_direction='both').clip(lower=0)
    elif method == 'linear':
        data_daily = data_daily.interpolate(method='linear', limit_direction='both')
    else:
        data_daily = data_daily.ffill().bfill()

    return data_daily


def create_sota_temporal_features(dates: pd.DatetimeIndex,
                                   sos_date=None, eos_date=None) -> np.ndarray:
    """
    Create Fourier-based temporal features for periodic pattern encoding.

    Replaced redundant season_sin/season_cos with crop-calendar-relative position.
    Previous columns 4-5 were duplicates of columns 2-3 (just month with different offset).
    Now columns 4-5 encode position relative to crop calendar (0=SOS, 1=EOS).

    Args:
        dates: DatetimeIndex to encode
        sos_date: Start of season date for relative position encoding
        eos_date: End of season date for relative position encoding

    Returns:
        Array of shape (len(dates), 6) with sin/cos encodings:
        - col 0-1: Day-of-year (annual cycle)
        - col 2-3: Month (coarser annual cycle)
        - col 4-5: Crop-calendar-relative position (or zeros if no calendar)
    """
    doy_norm = dates.dayofyear / 365.0
    month_norm = (dates.month - 1) / 12.0  # 0-indexed for consistency

    if sos_date is not None and eos_date is not None:
        # Crop-calendar-relative position: 0 at SOS, 1 at EOS
        # This is genuinely useful for agronomic modeling
        total_days = max((eos_date - sos_date).days, 1)
        rel_pos = np.clip(
            [(d - sos_date).days / total_days for d in dates], 0, 1
        )
        season_sin = np.sin(2 * np.pi * rel_pos)
        season_cos = np.cos(2 * np.pi * rel_pos)
    else:
        # No crop calendar available - use zeros
        season_sin = np.zeros(len(dates))
        season_cos = np.zeros(len(dates))

    return np.column_stack([
        np.sin(2 * np.pi * doy_norm),    # col 0: sin_doy
        np.cos(2 * np.pi * doy_norm),    # col 1: cos_doy
        np.sin(2 * np.pi * month_norm),  # col 2: sin_month
        np.cos(2 * np.pi * month_norm),  # col 3: cos_month
        season_sin,                       # col 4: season_sin (relative to crop calendar)
        season_cos,                       # col 5: season_cos (relative to crop calendar)
    ])

# % Feature engineering
def _get_aggregation_params(aggregation: str, year: int,
                             crop_season_info=None) -> Tuple[pd.DatetimeIndex, int, str]:
    """
    Return target DatetimeIndex, sequence length, and frequency string.

    Fragile period-then-filter approach replaced with date-range-first approach.
    Leap day filtering now happens before period trimming to prevent off-by-one errors.
    """
    freq_map = {"daily": (DAILY_FREQ, 365), "weekly": (WEEKLY_FREQ, 52), "dekad": (DEKAD_FREQ, 36)}
    if aggregation not in freq_map:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    freq_str, default_len = freq_map[aggregation]

    if crop_season_info is not None:
        cutoff_date = crop_season_info['cutoff_date']
        sos_date = crop_season_info.get('sos_date')
        if sos_date is not None:
            # Generate date range from SOS to cutoff, then filter leap days
            raw_dates = pd.date_range(start=sos_date, end=cutoff_date, freq=freq_str)
        else:
            # No SOS date, work backwards from cutoff
            raw_dates = pd.date_range(end=cutoff_date, periods=default_len + 5, freq=freq_str)
    else:
        # No crop season info, use year-end
        raw_dates = pd.date_range(end=f"{year}-12-31", periods=default_len + 5, freq=freq_str)

    # Filter leap days BEFORE trimming to prevent off-by-one
    target_dates = raw_dates[~((raw_dates.month == 2) & (raw_dates.day == 29))]

    # Trim to max allowed length from the END (most recent dates)
    target_dates = target_dates[-default_len:]

    # Validate: warn if we got significantly fewer dates than expected
    if len(target_dates) < default_len * 0.8:
        logging.warning(
            f"[{aggregation}] Year {year}: expected ~{default_len} periods, "
            f"got {len(target_dates)}. Check crop_season_info bounds."
        )

    return target_dates, len(target_dates), freq_str


def _extract_weather_features(dataset: CYDataset, adm_id: str, year: int,
                               target_dates: pd.DatetimeIndex, aggregation: str,
                               weather_features_list: List[str],
                               debug: bool = False) -> np.ndarray:
    """
    Extract and aggregate weather features.

    Now accepts weather_features_list parameter derived from config
    instead of using a hardcoded list, respecting use_cwb_feature and drop_tavg flags.

    Args:
        dataset: CY-Bench dataset
        adm_id: Administrative region ID
        year: Year to extract data for
        target_dates: DatetimeIndex for resampling
        aggregation: Temporal aggregation ('daily', 'weekly', 'dekad')
        weather_features_list: List of weather features to extract (from config.weather_features)
        debug: Enable debug logging (deprecated, use logging level)

    Returns:
        Array of shape (seq_len, len(weather_features_list)) with weather data
    """
    seq_len = len(target_dates)
    n_weather = len(weather_features_list)
    weather_features = np.zeros((seq_len, n_weather), dtype=np.float32)

    if "meteo" not in dataset._dfs_x:
        logging.warning(f"[{adm_id}] No meteorological data available")
        return weather_features

    try:
        meteo = dataset._dfs_x["meteo"].loc[adm_id]
        all_meteo = meteo.reset_index() if isinstance(meteo, pd.Series) else meteo
        year_data = (all_meteo[all_meteo[KEY_YEAR] == year]
                     if KEY_YEAR in all_meteo.columns else all_meteo).copy()
        if year_data.empty:
            logging.warning(f"[{adm_id}] No data for year {year}, using last 365 days")
            year_data = all_meteo.tail(365)

        # Iterate over config-derived feature list instead of hardcoded list
        if aggregation == "daily":
            # Build output with correct column order, filling missing columns with NaN
            # Then convert NaN to 0 (mean in z-score space) during normalization
            weather_features = np.full((seq_len, n_weather), np.nan, dtype=np.float32)
            daily_df = pd.DataFrame(index=target_dates)
            for j, col in enumerate(weather_features_list):
                if col in year_data.columns:
                    daily_df[col] = interpolate_to_daily(year_data[col], target_dates,
                                                         method='linear',
                                                         interpolate_data='weather')
                    weather_features[:, j] = daily_df[col].values
            # Leave NaNs in place - normalization will impute to 0.0 (mean in z-score space)
            # This ensures correct z-score computation instead of using raw 0.0 values
        else:
            # For weekly/dekad: interpolate to FULL daily resolution first, then aggregate
            # This ensures true temporal averaging instead of point-sampling
            freq = WEEKLY_FREQ if aggregation == "weekly" else DEKAD_FREQ
            full_daily_range = pd.date_range(start=target_dates[0], end=target_dates[-1], freq='D')

            daily_df_full = pd.DataFrame(index=full_daily_range)
            for col in weather_features_list:
                if col in year_data.columns:
                    daily_df_full[col] = interpolate_to_daily(year_data[col], full_daily_range,
                                                               method='linear',
                                                               interpolate_data='weather')

            # Now aggregate to target frequency (true temporal averaging)
            resampled = daily_df_full.resample(freq).mean()

            # Reindex to match exact target_dates
            expected_index = pd.date_range(start=target_dates[0], periods=seq_len, freq=freq)
            resampled = resampled.reindex(expected_index)

            # Build output with correct column order, filling missing columns with NaN
            weather_features = np.full((seq_len, n_weather), np.nan, dtype=np.float32)
            for j, col in enumerate(weather_features_list):
                if col in resampled.columns:
                    weather_features[:, j] = resampled[col].values
            # Leave NaNs in place - normalization will impute to 0.0 (mean in z-score space)
    except Exception as e:
        logging.error(f"[{adm_id}] Weather extraction error: {e}")

    return weather_features


def _extract_remote_sensing_features(dataset: CYDataset, adm_id: str, year: int,
                                     target_dates: pd.DatetimeIndex, aggregation: str,
                                     debug: bool = False) -> np.ndarray:
    """
    Extract and aggregate remote sensing features (fpar, ndvi, ssm, rsm).

    Args:
        dataset: CY-Bench dataset
        adm_id: Administrative region ID
        year: Year to extract data for
        target_dates: DatetimeIndex for resampling
        aggregation: Temporal aggregation ('daily', 'weekly', 'dekad')
        debug: Enable debug logging (deprecated, use logging level)

    Returns:
        Array of shape (seq_len, 4) with remote sensing features
    """
    seq_len = len(target_dates)
    rs_features = np.zeros((seq_len, 4))

    for i, rs_var in enumerate(["fpar", "ndvi", "ssm", "rsm"]):
        try:
            if rs_var in ["ssm", "rsm"]:
                if "soil_moisture" not in dataset._dfs_x:
                    continue
                df = dataset._dfs_x["soil_moisture"]
                if (adm_id, year) not in df.index:
                    continue
                rs_data = df.loc[(adm_id, year)].iloc[:, 0]
            else:
                if rs_var not in dataset._dfs_x:
                    continue
                df = dataset._dfs_x[rs_var]
                if (adm_id, year) not in df.index:
                    continue
                rs_data = df.loc[(adm_id, year)].iloc[:, 0]

            # For daily aggregation, interpolate directly to target_dates
            if aggregation == "daily":
                daily_val = interpolate_to_daily(rs_data, target_dates,
                                                 interpolate_data='soil_moisture' if rs_var in ['ssm', 'rsm'] else rs_var)
                rs_features[:, i] = daily_val.values
            else:
                # For weekly/dekad: interpolate to full daily range, then aggregate
                freq = WEEKLY_FREQ if aggregation == "weekly" else DEKAD_FREQ
                full_daily_range = pd.date_range(start=target_dates[0], end=target_dates[-1], freq='D')
                daily_val_full = interpolate_to_daily(rs_data, full_daily_range,
                                                      interpolate_data='soil_moisture' if rs_var in ['ssm', 'rsm'] else rs_var)
                # Aggregate to target frequency
                aggregated = pd.DataFrame({rs_var: daily_val_full}, index=full_daily_range).resample(freq).mean()
                # Reindex to match exact target_dates
                expected_index = pd.date_range(start=target_dates[0], periods=seq_len, freq=freq)
                aggregated = aggregated.reindex(expected_index)  # Leave NaNs, normalization handles them
                rs_features[:, i] = aggregated[rs_var].values
        except Exception as e:
            logging.warning(f"[{adm_id}] {rs_var} extraction error: {e}")

    return rs_features


def _extract_static_features(dataset: CYDataset, adm_id: str, year: int,
                              include_spatial_features: bool,
                              lat: Optional[float], lon: Optional[float],
                              lag_years: int,
                              daily_df: Optional[pd.DataFrame] = None,
                              debug: bool = False) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """
    Assemble the static feature vector for one location-year.

    Feature order (MUST match _get_static_feature_names() exactly):
      1. Soil properties
      2. Location properties  →  also extracts lat/lon as side-effect
      3. Crop calendar dates
      4. Explicit lat/lon      (conditional)
      5. Lagged yields         (conditional)

    Args:
        dataset: CY-Bench dataset
        adm_id: Administrative region ID
        year: Year to extract features for
        include_spatial_features: Whether to include explicit lat/lon features
        lat: Latitude hint (will be overwritten from location data)
        lon: Longitude hint (will be overwritten from location data)
        lag_years: Number of lagged yield features to include
        daily_df: Unused (kept for compatibility)
        debug: Enable debug logging (deprecated, use logging level)

    Returns:
        Tuple of (static_features_array, latitude, longitude)

    NOTE ON LAG YIELD LEAKAGE: In operational forecasting, the lag yield for year t
    is the *observed* yield from year t-1. For test year t, lag_yield_1 = y(t-1),
    which may itself be a test year. This is realistic for operational use
    (prior year yield is published before the current season ends) but means
    the model has indirect access to test-set yield magnitudes. This is documented
    here as an explicit assumption, not a bug.
    """
    static_vals = []

    # 1. Soil properties
    if "soil" in dataset._dfs_x:
        try:
            soil = dataset._dfs_x["soil"].loc[adm_id]
            for prop in SOIL_PROPERTIES:
                static_vals.append(float(soil.get(prop, np.nan)))
        except Exception as e:
            logging.warning(f"[{adm_id}] Soil extraction error: {e}")
            static_vals.extend([np.nan] * len(SOIL_PROPERTIES))
    else:
        static_vals.extend([np.nan] * len(SOIL_PROPERTIES))

    # 2. Location properties (also extracts lat/lon)
    if "location" in dataset._dfs_x:
        try:
            loc = dataset._dfs_x["location"].loc[adm_id]
            for prop in LOCATION_PROPERTIES:
                val = loc.get(prop, np.nan)
                static_vals.append(float(val) if val is not None else np.nan)
                if prop == "latitude":
                    lat = float(val) if val is not None else None
                elif prop == "longitude":
                    lon = float(val) if val is not None else None
        except Exception as e:
            logging.warning(f"[{adm_id}] Location extraction error: {e}")
            static_vals.extend([np.nan] * len(LOCATION_PROPERTIES))
    else:
        static_vals.extend([np.nan] * len(LOCATION_PROPERTIES))

    # 3. Crop calendar dates (with cyclic encoding for day-of-year features)
    if ("crop_season" in dataset._dfs_x and
            (adm_id, year) in dataset._dfs_x["crop_season"].index):
        crop = dataset._dfs_x["crop_season"].loc[(adm_id, year)]
    else:
        crop = pd.Series([np.nan] * 4, index=CROP_CALENDAR_DATES)

    for name, v in zip(CROP_CALENDAR_DATES, crop):
        if isinstance(v, pd.Timestamp):
            doy = float(v.dayofyear)
        elif v is not None:
            doy = float(v)
        else:
            doy = np.nan

        # Cyclic encoding for day-of-year features (sos_date, eos_date)
        # This ensures that DOY=365 and DOY=1 are treated as adjacent (1 day apart)
        if name in ["sos_date", "eos_date"]:
            if not np.isnan(doy):
                cyclic_val = 2 * np.pi * doy / 365.0
                static_vals.extend([np.sin(cyclic_val), np.cos(cyclic_val)])
            else:
                # Always append 2 values for cyclic features, even if NaN
                static_vals.extend([np.nan, np.nan])
        else:
            # cutoff_date and season_window_length are linear, not cyclic
            static_vals.append(doy)

    # 4. Explicit spatial features
    if include_spatial_features:
        static_vals.append(lat if lat is not None else np.nan)
        static_vals.append(lon if lon is not None else np.nan)

    # 5. Lagged yields
    for lag in range(1, lag_years + 1):
        lag_value = np.nan
        try:
            if hasattr(dataset, '_data_target'):
                v = dataset._data_target.loc[(adm_id, year - lag)]
                lag_value = float(v.iloc[0] if isinstance(v, pd.Series) else v)
            else:
                indices = list(dataset.indices())
                if (adm_id, year - lag) in indices:
                    targets = list(dataset.targets())
                    lag_value = float(targets[indices.index((adm_id, year - lag))])
        except (KeyError, IndexError, ValueError):
            pass
        # Impute missing lags to NaN; normalization will convert to 0.0 in z-score space
        static_vals.append(lag_value if not np.isnan(lag_value) else np.nan)

    if lag_years > 0:
        logging.debug(f"[{adm_id}] Added {lag_years} lagged yield features")

    return np.array(static_vals, dtype=np.float32), lat, lon


def _assemble_features(features: Dict, seq_len: int,
                       use_sota_features: bool,
                       weather_features_list: List[str]) -> np.ndarray:
    """
    Concatenate time series feature arrays in consistent column order.

    Now accepts weather_features_list parameter instead of using global.
    """
    n_weather = len(weather_features_list)
    n_rs = len(REMOTE_SENSING_FEATURES)      # 4
    n_sota = len(SOTA_TEMPORAL_VARS_LIST) if use_sota_features else 0

    X = np.zeros((seq_len, n_weather + n_rs + n_sota), dtype=np.float32)
    col = 0

    if 'weather' in features:
        # Assert exact shape match - _extract_weather_features now guarantees this
        assert features['weather'].shape == (seq_len, n_weather), (
            f"Weather shape mismatch: expected ({seq_len}, {n_weather}), "
            f"got {features['weather'].shape}. This indicates a bug in "
            "_extract_weather_features - it should always return n_weather columns."
        )
        X[:, col:col + n_weather] = features['weather']
    col += n_weather

    if 'remote_sensing' in features:
        assert features['remote_sensing'].shape == (seq_len, n_rs)
        X[:, col:col + n_rs] = features['remote_sensing']
    col += n_rs

    if use_sota_features and 'sota_temporal' in features:
        assert features['sota_temporal'].shape == (seq_len, n_sota)
        X[:, col:col + n_sota] = features['sota_temporal']

    return X

def build_daily_input_sequence(
        dataset: CYDataset, adm_id: str, year: int,
        aggregation: str = "dekad",
        use_sota_features: bool = False,
        include_spatial_features: bool = False,
        lag_years: int = 0,
        weather_features_list: Optional[List[str]] = None,
        debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, Dict, np.ndarray]:
    """
    Build model-ready input for one location-year.

    Args:
        dataset: CY-Bench dataset
        adm_id: Administrative region ID
        year: Year to extract data for
        aggregation: Temporal aggregation ('daily', 'weekly', 'dekad')
        use_sota_features: Include SOTA temporal features
        include_spatial_features: Include explicit lat/lon features
        lag_years: Number of lagged yield features (max 2)
        weather_features_list: List of weather features to extract (from config.weather_features)
        debug: Enable debug logging (deprecated, use logging level)

    Returns:
        X_ts: Time series features of shape (seq_len, n_ts_features)
        X_static: Static features of shape (n_static_features,)
        y: Target yield (original scale)
        meta: Dictionary with adm_id, year, lat, lon, and shapes
        validity_mask: Boolean array of shape (seq_len,) for data validity
    """
    # Default to base weather features if not specified
    if weather_features_list is None:
        weather_features_list = WEATHER_FEATURES_BASE

    logging.debug(f"Building sequence: {adm_id}, {year}, {aggregation}")

    # Crop season trimming
    crop_season_info = None
    if ("crop_season" in dataset._dfs_x and
            (adm_id, year) in dataset._dfs_x["crop_season"].index):
        crop_season_info = dataset._dfs_x["crop_season"].loc[(adm_id, year)]

    target_dates, seq_len, freq_str = _get_aggregation_params(aggregation, year, crop_season_info)

    features = {}
    # Pass weather_features_list to extraction function
    features['weather'] = _extract_weather_features(
        dataset, adm_id, year, target_dates, aggregation, weather_features_list, debug)
    features['remote_sensing'] = _extract_remote_sensing_features(
        dataset, adm_id, year, target_dates, aggregation, debug)

    # SOTA temporal features with crop-calendar-relative position
    # Pass sos_date and eos_date from crop_season_info for meaningful season-relative features
    if use_sota_features:
        sos_date = crop_season_info.get('sos_date') if crop_season_info is not None else None
        eos_date = crop_season_info.get('eos_date') if crop_season_info is not None else None
        sota = create_sota_temporal_features(target_dates, sos_date=sos_date, eos_date=eos_date)
        if aggregation == "daily":
            features['sota_temporal'] = sota
        else:
            freq = WEEKLY_FREQ if aggregation == "weekly" else DEKAD_FREQ
            # Resample and reindex to ensure exact shape match
            sota_df = pd.DataFrame(sota, index=target_dates)
            resampled = sota_df.resample(freq).mean()
            expected_index = pd.date_range(start=target_dates[0], periods=len(target_dates), freq=freq)
            resampled = resampled.reindex(expected_index)  # Leave NaNs, normalization handles them
            features['sota_temporal'] = resampled.values

    # Static features
    features['static'], lat, lon = _extract_static_features(
        dataset, adm_id, year, include_spatial_features, None, None,
        lag_years,
        daily_df=None,  # No GDD growth stage features
        debug=debug,
    )

    # Target yield
    try:
        if hasattr(dataset, '_data_target'):
            v = dataset._data_target.loc[(adm_id, year)]
            y = float(v.iloc[0] if isinstance(v, pd.Series) else v)
        else:
            indices = list(dataset.indices())
            targets = list(dataset.targets())
            y = float(targets[indices.index((adm_id, year))])
    except Exception as e:
        logging.warning(f"[{adm_id}] Target extraction error: {e}")
        y = 0.0

    # Pass weather_features_list to _assemble_features
    X_ts = _assemble_features(features, seq_len, use_sota_features, weather_features_list)
    X_static = features['static'].astype(np.float32)

    logging.debug(f"[{adm_id}] X_ts={X_ts.shape}, X_static={X_static.shape}")

    # --- Pad and mask to handle variable sequence lengths ---
    max_len = MAX_SEQ_LENS[aggregation]
    actual_len = X_ts.shape[0]

    if actual_len >= max_len:
        # Truncate if longer than max (shouldn't happen, but safety check)
        X_ts_out = X_ts[:max_len]
        observed_mask = np.ones(max_len, dtype=bool)
    else:
        # Pad with zeros and create mask
        pad_len = max_len - actual_len
        X_ts_out = np.concatenate([
            X_ts,
            np.zeros((pad_len, X_ts.shape[1]), dtype=np.float32)
        ], axis=0)
        observed_mask = np.concatenate([
            np.ones(actual_len, dtype=bool),
            np.zeros(pad_len, dtype=bool)
        ])

    meta = {"adm_id": adm_id, "year": year, "lat": lat, "lon": lon,
            "seq_len": actual_len, "padded_len": X_ts_out.shape[0],
            "n_ts": X_ts_out.shape[1], "n_static": X_static.shape[0]}

    # validity_mask is now the observed_mask for compatibility
    validity_mask = observed_mask

    return X_ts_out, X_static, y, meta, validity_mask

# %% Dataset Wrapper
class DailyYieldDataset(Dataset):
    """PyTorch Dataset wrapping pre-computed arrays for one data split."""

    def __init__(self, X_ts, X_static, y, years=None, adm_ids=None,
                 lats=None, lons=None, validity_masks=None):
        self.X_ts = torch.tensor(X_ts, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.years = torch.tensor(years, dtype=torch.long) if years is not None else None
        self.adm_ids = list(adm_ids) if adm_ids is not None else None

        # replace None with nan before passing to torch.tensor().
        # _extract_static_features() can return lat=None / lon=None when location
        # data is missing. torch.tensor([1.2, None, 3.4]) raises a TypeError.
        def _safe_tensor(lst):
            if lst is None:
                return None
            cleaned = [v if v is not None else float('nan') for v in lst]
            return torch.tensor(cleaned, dtype=torch.float32)

        self.lats = _safe_tensor(lats)
        self.lons = _safe_tensor(lons)
        self.validity_masks = (torch.tensor(validity_masks, dtype=torch.bool)
                               if validity_masks is not None else None)

    def __len__(self):
        return len(self.X_ts)

    def __getitem__(self, idx):
        vm = (self.validity_masks[idx] if self.validity_masks is not None
              else torch.ones(self.X_ts.shape[1], dtype=torch.bool))
        return (self.X_ts[idx], self.X_static[idx], self.y[idx],
                self.years[idx], self.adm_ids[idx],
                self.lats[idx], self.lons[idx], vm)
    
# %% Data Module
class DailyCYBenchSeqDataModule(pl.LightningDataModule):
    """Lightning DataModule: loads CY-Bench data, builds features, normalises."""

    def __init__(self, config: Union[TSTModelConfig, LinearModelConfig]):
        super().__init__()
        # Ignore config in save_hyperparameters to avoid circular checkpoint references
        # (TSTModelConfig or LinearModelConfig contains load_checkpoint which could point to another checkpoint)
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        self.y_mean = self.y_std = None
        self.train_ds = self.val_ds = self.test_ds = None
        self.feature_norm_params = None
        # Store all data for dynamic splitting
        self.all_X_ts = None
        self.all_X_static = None
        self.all_y = None
        self.all_years = None
        self.all_adm_ids = None
        self.all_lats = None
        self.all_lons = None
        self.all_masks = None
        # Recursive lag prediction cache
        self._prediction_cache = {}  # {(adm_id, year): predicted_yield}
        self._test_years = None  # Track which years are in test set
        self._train_years = None  # Track which years are in train set

    def setup(self, stage: Optional[str] = None,
              train_years: Optional[List[int]] = None,
              val_years: Optional[List[int]] = None,
              test_years: Optional[List[int]] = None,
              features_only: bool = False):
        """
        Setup datasets and features.

        Args:
            stage: 'fit', 'validate', 'test', or None
            train_years: Years for training split
            val_years: Years for validation split
            test_years: Years for test split
            features_only: If True, only build feature arrays and skip split/normalization
                          (useful for caching features across CV folds)
        """
        cfg = self.config
        # If called by Lightning internals during validate/test with no explicit splits,
        # and we already have datasets, skip re-setup to preserve the current fold split.
        if (stage in ('validate', 'test') and
            train_years is None and
            self.train_ds is not None):
            return

        # Always recompute normalization for the current split
        # Prevents stale params from a previous setup() call leaking into a new split
        self.feature_norm_params = None
        self.y_mean = None
        self.y_std = None

        print(f"\n[DataModule] {cfg.crop}-{cfg.country} | {cfg.aggregation.upper()} | "
              f"Spatial={cfg.include_spatial_features} | Lag={cfg.lag_years}")

        df_y, dfs_x = load_dfs_crop(cfg.crop, [cfg.country])
        if df_y is None or len(df_y) == 0:
            raise ValueError(f"No data for {cfg.crop}-{cfg.country}")

        ds = CYDataset(cfg.crop, df_y, dfs_x)

        # Build all features (only done once)
        if self.all_X_ts is None:
            all_X_ts, all_X_static, all_y = [], [], []
            all_years_list, all_adm_ids, all_lats, all_lons, all_masks = [], [], [], [], []

            for i in range(len(ds)):
                sample = ds[i]
                X_ts, X_static, y, meta, mask = build_daily_input_sequence(
                    ds, sample[KEY_LOC], sample[KEY_YEAR],
                    aggregation=cfg.aggregation,
                    use_sota_features=cfg.use_sota_features,
                    include_spatial_features=cfg.include_spatial_features,
                    lag_years=cfg.lag_years,
                    weather_features_list=cfg.weather_features,  # Pass from config
                )
                all_X_ts.append(X_ts)
                all_X_static.append(X_static)
                all_y.append(y)
                all_years_list.append(sample[KEY_YEAR])
                all_adm_ids.append(sample[KEY_LOC])
                all_lats.append(meta["lat"])
                all_lons.append(meta["lon"])
                all_masks.append(mask)

            # Convert to numpy arrays
            self.all_X_ts = np.array(all_X_ts)
            self.all_X_static = np.array(all_X_static)
            self.all_y = np.array(all_y)
            self.all_years = np.array(all_years_list)
            self.all_adm_ids = np.array(all_adm_ids)
            self.all_lats = np.array(all_lats, dtype=object)
            self.all_lons = np.array(all_lons, dtype=object)
            self.all_masks = np.array(all_masks)

            # Validate static feature count
            expected = cfg._compute_expected_static_features()
            actual = self.all_X_static.shape[1]
            if actual != expected:
                raise ValueError(
                    f"\n[ERROR] Static feature mismatch: expected {expected}, got {actual}.\n"
                    f"A feature creation step likely failed silently. Check debug logs.\n"
                    f"Config: spatial={cfg.include_spatial_features}, lag={cfg.lag_years}"
                )
            print(f"  Static features validated: {actual}/{expected}")

        # Early return for features-only mode (avoids wasteful split/normalization)
        # Moved OUTSIDE the `if self.all_X_ts is None` block so it works even when arrays are cached
        if features_only:
            logging.info(f"[DataModule] features_only=True - skipping split/normalization, "
                       f"cached {len(self.all_X_ts)} samples")
            return

        # Use provided splits or compute default
        if train_years is not None and val_years is not None and test_years is not None:
            # Dynamic splits provided
            train_yrs = set(train_years)
            val_yrs = set(val_years)
            test_yrs = set(test_years)
        else:
            # Default split for backward compatibility
            years_sorted = np.unique(self.all_years)
            if len(years_sorted) < 6:
                raise ValueError(
                    f"Need ≥ 6 years for split (3 test + 3 val + train); got {len(years_sorted)}: {sorted(years_sorted)}"
                )
            test_yrs = set(years_sorted[-3:])
            val_yrs = set(years_sorted[-6:-3])
            train_yrs = set(years_sorted[:-6])

        print(f"  Split: Train {sorted(train_yrs)}, Val {sorted(val_yrs)}, Test {sorted(test_yrs)}")
        print(f"  Years Summary:")
        print(f"    Train years ({len(train_yrs)}): {sorted(train_yrs)}")
        print(f"    Val years ({len(val_yrs)}):   {sorted(val_yrs)}")
        print(f"    Test years ({len(test_yrs)}):  {sorted(test_yrs)}")

        def idx(yr_set):
            return np.where(np.isin(self.all_years, list(yr_set)))[0]

        train_idx, val_idx, test_idx = idx(train_yrs), idx(val_yrs), idx(test_yrs)

        # Normalise targets using training statistics only (no leakage)
        self.y_mean = float(np.mean(self.all_y[train_idx]))
        self.y_std = float(np.std(self.all_y[train_idx])) or 1.0
        print(f"  Y norm: mean={self.y_mean:.4f}, std={self.y_std:.4f}")

        self.feature_norm_params = self._compute_feature_normalization(
            self.all_X_ts[train_idx], self.all_X_static[train_idx],
            self.all_masks[train_idx] if self.all_masks is not None else None
        )
        print(f"  Feature norm params: {len(self.feature_norm_params)} features")

        all_y_norm = (self.all_y - self.y_mean) / self.y_std

        def make_ds(idxs):
            return DailyYieldDataset(
                self.all_X_ts[idxs], self.all_X_static[idxs], all_y_norm[idxs],
                self.all_years[idxs], self.all_adm_ids[idxs],
                self.all_lats[idxs], self.all_lons[idxs], self.all_masks[idxs],
            )

        self.train_ds = make_ds(train_idx)
        self.val_ds = make_ds(val_idx)
        self.test_ds = make_ds(test_idx)
        print(f"  Samples: train={len(self.train_ds)}, val={len(self.val_ds)}, "
              f"test={len(self.test_ds)}")

        # Store train/test years for recursive lag prediction
        self._train_years = train_yrs
        self._test_years = test_yrs
        self._val_years = val_yrs

        # Warn about potential lag yield leakage in test set
        if self.config.lag_years > 0:
            test_year_set = test_yrs
            lag_overlap_years = set()
            for test_year in test_yrs:
                for lag in range(1, self.config.lag_years + 1):
                    if (test_year - lag) in test_year_set:
                        lag_overlap_years.add(test_year - lag)
            if lag_overlap_years:
                msg = (
                    f"[WARNING] Lag Leakage: Test years {sorted(lag_overlap_years)} are used as lag "
                    f"inputs for later test years. Reported test metrics may be optimistic because "
                    f"the model has access to observed test-set yields. "
                    f"Use --lag_years 0 for no lag features, or --use_recursive_lags for true "
                    f"out-of-sample evaluation (uses predicted yields as lags)."
                )
                print(msg)
                logging.warning(msg)

    def _compute_feature_normalization(self, X_ts, X_static, observed_masks=None):
        """Compute z-score params from training data only.

        Args:
            X_ts: Time series features (n_samples, seq_len, n_features)
            X_static: Static features (n_samples, n_static_features)
            observed_masks: Boolean masks (n_samples, seq_len) indicating valid timesteps
        """
        params = {}
        for i, name in enumerate(self._get_ts_feature_names()):
            col = X_ts[:, :, i].flatten()
            # Exclude padded zeros if masks provided
            if observed_masks is not None:
                col = X_ts[:, :, i][observed_masks]
            params[f"ts_{name}"] = {
                "mean": float(np.nanmean(col)) if col.size > 0 and not np.all(np.isnan(col)) else 0.0,
                "std": (float(np.nanstd(col)) or 1.0) if col.size > 0 and not np.all(np.isnan(col)) else 1.0,
            }
        for i, name in enumerate(self._get_static_feature_names()):
            col = X_static[:, i]
            params[f"static_{name}"] = {
                "mean": float(np.nanmean(col)) if col.size > 0 and not np.all(np.isnan(col)) else 0.0,
                "std": (float(np.nanstd(col)) or 1.0) if col.size > 0 and not np.all(np.isnan(col)) else 1.0,
            }
        return params

    def _get_ts_feature_names(self) -> List[str]:
        # Use config.weather_features instead of global WEATHER_FEATURES
        names = [f'weather_{n}' for n in self.config.weather_features]
        names += [f'rs_{n}' for n in REMOTE_SENSING_FEATURES]
        if self.config.use_sota_features:
            names += [f'sota_{n}' for n in SOTA_TEMPORAL_VARS_LIST]
        return names

    def _get_static_feature_names(self) -> List[str]:
        """Thin wrapper around the module-level helper (single source of truth)."""
        return _get_static_feature_names(
            self.config.include_spatial_features,
            self.config.lag_years,
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.config.batch_size,
                          shuffle=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.config.batch_size,
                          shuffle=False, num_workers=self.config.num_workers)

    def test_dataloader(self):
        """
        Returns the test dataloader.

        NOTE: shuffle=False processes samples in dataset construction order.
        For --use_recursive_lags to work correctly, samples within each location must
        appear in chronological order. This holds as long as ds[i] iterates chronologically
        within each location, which is assumed but not enforced by CYDataset.
        """
        return DataLoader(self.test_ds, batch_size=self.config.batch_size,
                          shuffle=False, num_workers=self.config.num_workers)
    

def calculate_fixed_split(
    all_years: List[int],
    test_years: int = 5,
    val_years: int = 2
) -> Dict:
    """
    Calculate fixed train/val/test splits for non-CV mode.
    
    Uses last N years for test set, last M years of remaining for validation,
    and everything else for training.
    """
    sorted_years = sorted(all_years)
    total_years = len(sorted_years)
    
    min_required = test_years + val_years + 1
    if total_years < min_required:
        raise ValueError(
            f"Insufficient data for fixed split mode: {total_years} years available, "
            f"but {min_required} years required (test={test_years} + val={val_years} + min_train=1). "
            f"Please use a country-crop combination with at least {min_required} years of data."
        )
    
    test_years_list = sorted_years[-test_years:]
    remaining = sorted_years[:-test_years]
    val_years_list = remaining[-val_years:]
    train_years_list = remaining[:-val_years]
    
    return {
        'train_years': train_years_list,
        'val_years': val_years_list,
        'test_years': test_years_list,
        'total_years': total_years,
        'can_split': True,
        'skip_reason': None
    }