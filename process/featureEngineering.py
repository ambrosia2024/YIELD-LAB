import logging

import numpy as np
import pandas as pd

from typing import Optional, Dict, List, Tuple

from cybench.datasets.dataset import Dataset as CYDataset

from cybench.config import (
    GDD_BASE_TEMP, GDD_UPPER_LIMIT, LOCATION_PROPERTIES, SOIL_PROPERTIES,
    FORECAST_LEAD_TIME, KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, KEY_CROP_SEASON,
    CROP_CALENDAR_DATES
)

#%% Global constants
# Maximum sequence lengths for padding — ensures uniform tensor shapes
MAX_SEQ_LENS = {"daily": 365, "weekly": 52, "dekad": 36}

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

DEKAD_FREQ = "10D"
WEEKLY_FREQ = "W-MON"
DAILY_FREQ = "D"

REMOTE_SENSING_FEATURES = ['fpar', 'ndvi', 'ssm', 'rsm']
WEATHER_FEATURES_BASE = ['tmin', 'tmax', 'tavg', 'prec', 'rad']

SOTA_TEMPORAL_VARS_LIST = [
    'sin_doy', 'cos_doy',
    'sin_month', 'cos_month',
    'season_sin', 'season_cos'
]

print(f"[Feature Config] SOTA Temporal vars ({len(SOTA_TEMPORAL_VARS_LIST)}): {SOTA_TEMPORAL_VARS_LIST}")

# %% Helper functions
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

def _compute_gdd_series(tavg: np.ndarray, tbase: float,
                        tupper: float) -> np.ndarray:
    """
    Compute daily Growing Degree Days (GDD) with upper threshold cap.

    Formula: GDD = clip(min(Tavg, Tupper) - Tbase, 0, None)

    Using min(Tavg, Tupper) instead of raw Tavg prevents days above the
    upper threshold from contributing more GDD than the optimal temperature,
    which is physiologically incorrect (development slows above Tupper).

    Args:
        tavg: Array of mean daily temperatures (°C), shape (seq_len,)
        tbase: Base temperature below which development stops (°C)
               Typically 8°C for maize, 5°C for wheat
        tupper: Upper threshold above which development stops (°C)
                Typically 30°C for maize, 25°C for wheat

    Returns:
        Array of daily GDD values, shape (seq_len,)
    """
    return np.maximum(np.minimum(tavg, tupper) - tbase, 0.0)

def _compute_rue_series(tavg: np.ndarray, cum_prec: np.ndarray,
                        cum_rad: np.ndarray, crop: str) -> np.ndarray:
    """
    Compute Radiation Use Efficiency (RUE) index time series.

    RUE_index = cumPAR * T_stress * W_stress

    Components:
      - cumPAR: cumulative photosynthetically active radiation (0.48 * cum_rad)
        0.48 is the standard PAR fraction of total shortwave radiation
        Reference: https://doi.org/10.1016/j.jag.2022.102724
      - T_stress: Gaussian temperature stress centered at crop-specific Topt
        Approximates the bell-shaped temperature response of photosynthesis
      - W_stress: Michaelis-Menten water availability term
        Saturating form: P/(P+300) where 300mm is the half-saturation constant

    Args:
        tavg: Mean daily temperature array (°C), shape (seq_len,)
        cum_prec: Cumulative precipitation array (mm), shape (seq_len,)
        cum_rad: Cumulative radiation array (MJ/m²), shape (seq_len,)
        crop: Crop name for Topt selection ('maize' or other)

    Returns:
        Array of RUE index values, shape (seq_len,)
    """
    Topt = 25.0 if crop == 'maize' else 20.0
    sigma = 7.0  # temperature sensitivity width (°C)

    cum_PAR = 0.48 * cum_rad
    T_stress = np.exp(-((tavg - Topt) ** 2) / (2 * sigma ** 2))
    W_stress = cum_prec / (cum_prec + 300.0 + 1e-9)

    rue = cum_PAR * T_stress * W_stress
    return np.nan_to_num(rue, nan=0.0, posinf=0.0, neginf=0.0)

def _arrhenius_response(tleaf: np.ndarray,
                        ea: float = 60000.0,
                        tref_k: float = 298.15,
                        r: float = 8.314) -> np.ndarray:
    """
    Arrhenius temperature response function.

    Computes the ratio k(T)/k(Tref) where k is a biochemical rate constant.
    Used to scale Rubisco carboxylation (Vcmax) and electron transport (Jmax)
    from reference temperature (25°C) to actual leaf temperature.

    Reference: Bernacchi et al. (2002) Plant Physiology 130:1992–1998
               https://doi.org/10.1104/pp.008250

    Args:
        tleaf: Leaf/air temperature (°C), shape (seq_len,)
        ea: Activation energy (J/mol), default 60000 J/mol
        tref_k: Reference temperature in Kelvin (default 298.15 = 25°C)
        r: Universal gas constant (J/mol/K)

    Returns:
        Dimensionless temperature response ratio, shape (seq_len,)
    """
    tk = tleaf + 273.15
    return np.exp((ea / r) * (1.0 / tref_k - 1.0 / tk))

def _compute_farquhar_series(tavg: np.ndarray, cum_prec: np.ndarray,
                             cum_rad: np.ndarray,
                             n: float = 0.1,
                             co2: float = 400.0) -> np.ndarray:
    """
    Compute Farquhar-von Caemmerer-Berry (FvCB) photosynthesis proxy time series.

    Approximates seasonal-scale net assimilation using the FvCB C3 model
    (Farquhar, von Caemmerer & Berry, 1980, Planta 149:78–90).

    NOTE ON SCALE LIMITATION: The FvCB model was derived for instantaneous
    leaf-scale processes. Applying it at dekadal/weekly/daily crop-season scale
    is a proxy approximation, not a mechanistic simulation. Results should be
    interpreted as a biophysically-motivated index, not true assimilation rates.

    Model components:
      Ac = Vcmax * (ci - Gamma) / (ci + Kc)     [Rubisco-limited]
      Aj = J * (ci - Gamma) / (4*ci + 8*Gamma)  [RuBP-limited]
      A  = harmonic mean(Ac, Aj)                 [co-limitation]

    Constants from Bernacchi et al. (2002):
      Kc = 404.9 µmol/mol (Michaelis constant for CO2)
      Gamma = 42.75 µmol/mol (CO2 compensation point)

    Args:
        tavg: Mean temperature array (°C), shape (seq_len,)
        cum_prec: Cumulative precipitation (mm), shape (seq_len,)
        cum_rad: Cumulative radiation (MJ/m²), shape (seq_len,)
        n: Leaf nitrogen proxy (dimensionless, default 0.1)
        co2: Ambient CO2 concentration (ppm, default 400.0)

    Returns:
        Array of FvCB proxy values, shape (seq_len,)
    """
    PAR = 0.48 * cum_rad

    temp_resp = _arrhenius_response(tavg)

    # Vcmax scales with leaf nitrogen (Rubisco is nitrogen-rich)
    # References: Evans (1989) Oecologia 78:9-19,
    #             Farquhar (1980) Planta 149:78-90
    Vcmax = 100.0 * temp_resp * (1.0 + 5.0 * n)

    # Electron transport capacity scales with PAR
    J = 0.1 * PAR * temp_resp

    # Intercellular CO2 under moderate water stress (C3 plants)
    # ci/ca ≈ 0.7 for well-watered C3, lower under drought
    # Reference: Wong et al. (1979) Plant Physiology 78:821-825
    ci = 0.7 * co2

    # FvCB biochemical constants (Bernacchi et al. 2002)
    Kc = 404.9      # Michaelis constant for CO2 (µmol/mol)
    Gamma = 42.75   # CO2 compensation point (µmol/mol)

    Ac = Vcmax * (ci - Gamma) / (ci + Kc + 1e-9)
    Aj = (J * (ci - Gamma)) / (4.0 * ci + 8.0 * Gamma + 1e-9)

    # Harmonic-like co-limitation (both limitations simultaneously active)
    A_proxy = (2.0 * Ac * Aj) / (Ac + Aj + 1e-9)

    # Water limitation: Michaelis-Menten form
    water_lim = cum_prec / (cum_prec + 300.0 + 1e-9)

    # Scale by PAR availability and water stress
    A_final = A_proxy * (PAR / (PAR + 100.0 + 1e-9)) * water_lim

    return np.nan_to_num(A_final, nan=0.0, posinf=0.0, neginf=0.0)

def _compute_heat_stress_counts(tavg_series: np.ndarray,
                                tmin_series: np.ndarray,
                                tmax_series: np.ndarray,
                                prec_series: np.ndarray,
                                crop: str,
                                validity_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute season-level threshold exceedance counts as static scalar features.

    These capture nonlinear biological responses that cumulative averages miss.
    A single heat stress day during pollination can cause irreversible yield loss
    (Schlenker & Roberts, 2009, PNAS 106:15594–15598).

    Features computed:
      - heat_stress_days:  days with Tmax > threshold (35°C maize, 30°C wheat)
      - frost_days:        days with Tmin < 0°C
      - cold_stress_days:  days with Tmin < 5°C
      - dry_days:          days with Prec < 1mm
      - wet_days:          days with Prec > 10mm
      - heat_stress_frac:  heat_stress_days / valid_days (0–1)
      - dry_frac:          dry_days / valid_days (0–1)

    Fractions are included alongside counts because they are invariant to
    season length, making them more comparable across crops and geographies.

    Args:
        tavg_series: Daily mean temperature (°C), shape (seq_len,)
        tmin_series: Daily minimum temperature (°C), shape (seq_len,)
        tmax_series: Daily maximum temperature (°C), shape (seq_len,)
        prec_series: Daily precipitation (mm), shape (seq_len,)
        crop: Crop name for threshold selection
        validity_mask: Boolean mask of valid (non-padded) timesteps,
                       shape (seq_len,). Only valid timesteps are counted.

    Returns:
        Dict mapping feature name → scalar float value
    """
    # Crop-specific heat stress threshold
    # Maize: 35°C (pollination failure threshold, Schlenker & Roberts 2009)
    # Wheat and others: 30°C (grain filling threshold)
    heat_thresh = 35.0 if crop == 'maize' else 30.0

    # Apply validity mask — only count real growing season days
    mask = validity_mask.astype(bool)
    valid_days = max(mask.sum(), 1)  # avoid division by zero

    tmax_v = tmax_series[mask]
    tmin_v = tmin_series[mask]
    prec_v = prec_series[mask]

    heat_stress = float(np.sum(tmax_v > heat_thresh))
    frost = float(np.sum(tmin_v < 0.0))
    cold_stress = float(np.sum(tmin_v < 5.0))
    dry = float(np.sum(prec_v < 1.0))
    wet = float(np.sum(prec_v > 10.0))

    return {
        'heat_stress_days': heat_stress,
        'frost_days': frost,
        'cold_stress_days': cold_stress,
        'dry_days': dry,
        'wet_days': wet,
        'heat_stress_frac': heat_stress / valid_days,
        'dry_frac': dry / valid_days,
    }

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
                               debug: bool = False,
                               # New parameters for domain features:
                               use_gdd: bool = False,
                               use_rue: bool = False,
                               use_farquhar: bool = False,
                               crop: str = 'maize',
                               gdd_base: float = 8.0,
                               gdd_upper: float = 30.0,
                               ) -> Dict[str, np.ndarray]:
    """
    Extract and aggregate weather features, optionally with domain features.

    Now accepts weather_features_list parameter derived from config
    instead of using a hardcoded list, respecting use_cwb_feature and drop_tavg flags.

    Now returns a dict with weather features plus raw daily arrays for
    heat stress computation, instead of just a single array.

    Args:
        dataset: CY-Bench dataset
        adm_id: Administrative region ID
        year: Year to extract data for
        target_dates: DatetimeIndex for resampling
        aggregation: Temporal aggregation ('daily', 'weekly', 'dekad')
        weather_features_list: List of weather features to extract (from config.weather_features)
        debug: Enable debug logging (deprecated, use logging level)
        use_gdd: Add cumulative GDD as a time series channel
        use_rue: Add RUE index as a time series channel
        use_farquhar: Add Farquhar proxy as a time series channel
        crop: Crop name for crop-specific parameters
        gdd_base: Base temperature for GDD computation
        gdd_upper: Upper threshold for GDD computation

    Returns:
        Dict with keys:
          - 'weather': Array of shape (seq_len, n_weather + n_domain) with all features
          - 'tavg_raw': Raw daily tavg array for heat stress (shape: seq_len or full_daily_len)
          - 'tmin_raw': Raw daily tmin array for heat stress
          - 'tmax_raw': Raw daily tmax array for heat stress
          - 'prec_raw': Raw daily prec array for heat stress
    """
    seq_len = len(target_dates)
    n_weather = len(weather_features_list)
    n_domain = sum([use_gdd, use_rue, use_farquhar])
    n_total = n_weather + n_domain

    # Initialize return dict with defaults
    result = {
        'weather': np.zeros((seq_len, n_total), dtype=np.float32),
        'tavg_raw': None,
        'tmin_raw': None,
        'tmax_raw': None,
        'prec_raw': None,
    }

    if "meteo" not in dataset._dfs_x:
        logging.warning(f"[{adm_id}] No meteorological data available")
        return result

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
            result['weather'] = np.full((seq_len, n_total), np.nan, dtype=np.float32)
            daily_df = pd.DataFrame(index=target_dates)

            # Extract base weather features
            for j, col in enumerate(weather_features_list):
                if col in year_data.columns:
                    daily_df[col] = interpolate_to_daily(year_data[col], target_dates,
                                                         method='linear',
                                                         interpolate_data='weather')
                    result['weather'][:, j] = daily_df[col].values

            # Extract raw arrays needed for domain features and heat stress
            for needed_col in ['tavg', 'tmin', 'tmax', 'prec', 'rad']:
                if needed_col not in daily_df.columns and needed_col in year_data.columns:
                    daily_df[needed_col] = interpolate_to_daily(
                        year_data[needed_col], target_dates,
                        method='linear', interpolate_data='weather')

            # Store raw arrays for heat stress (these are already at target resolution)
            result['tavg_raw'] = daily_df['tavg'].values if 'tavg' in daily_df else np.zeros(seq_len)
            result['tmin_raw'] = daily_df['tmin'].values if 'tmin' in daily_df else np.zeros(seq_len)
            result['tmax_raw'] = daily_df['tmax'].values if 'tmax' in daily_df else np.zeros(seq_len)
            result['prec_raw'] = daily_df['prec'].values if 'prec' in daily_df else np.zeros(seq_len)

            # Domain feature channels (appended after base weather)
            col_idx = n_weather
            if use_gdd:
                gdd = _compute_gdd_series(result['tavg_raw'], gdd_base, gdd_upper)
                cum_gdd = np.cumsum(gdd)
                result['weather'][:, col_idx] = cum_gdd.astype(np.float32)
                col_idx += 1

            if use_rue or use_farquhar:
                # Need cumulative prec and rad
                cum_prec = np.nancumsum(result['prec_raw'])
                rad_raw = daily_df['rad'].values if 'rad' in daily_df else np.zeros(seq_len)
                cum_rad = np.nancumsum(rad_raw)

                if use_rue:
                    rue = _compute_rue_series(result['tavg_raw'], cum_prec, cum_rad, crop)
                    result['weather'][:, col_idx] = rue.astype(np.float32)
                    col_idx += 1

                if use_farquhar:
                    farq = _compute_farquhar_series(result['tavg_raw'], cum_prec, cum_rad)
                    result['weather'][:, col_idx] = farq.astype(np.float32)

            # Leave NaNs in place - normalization will impute to 0.0 (mean in z-score space)
            # This ensures correct z-score computation instead of using raw 0.0 values
        else:
            # For weekly/dekad: interpolate to FULL daily resolution first, then aggregate
            # This ensures true temporal averaging instead of point-sampling
            freq = WEEKLY_FREQ if aggregation == "weekly" else DEKAD_FREQ
            full_daily_range = pd.date_range(start=target_dates[0], end=target_dates[-1], freq='D')

            daily_df_full = pd.DataFrame(index=full_daily_range)

            # Extract base weather features to daily resolution
            for col in weather_features_list:
                if col in year_data.columns:
                    daily_df_full[col] = interpolate_to_daily(year_data[col], full_daily_range,
                                                               method='linear',
                                                               interpolate_data='weather')

            # Also extract raw arrays needed for domain features and heat stress
            for needed_col in ['tavg', 'tmin', 'tmax', 'prec', 'rad']:
                if needed_col not in daily_df_full.columns and needed_col in year_data.columns:
                    daily_df_full[needed_col] = interpolate_to_daily(
                        year_data[needed_col], full_daily_range,
                        method='linear', interpolate_data='weather')

            # Store raw daily arrays for heat stress computation
            # (heat stress uses daily data regardless of aggregation)
            result['tavg_raw'] = daily_df_full['tavg'].values if 'tavg' in daily_df_full else np.zeros(len(full_daily_range))
            result['tmin_raw'] = daily_df_full['tmin'].values if 'tmin' in daily_df_full else np.zeros(len(full_daily_range))
            result['tmax_raw'] = daily_df_full['tmax'].values if 'tmax' in daily_df_full else np.zeros(len(full_daily_range))
            result['prec_raw'] = daily_df_full['prec'].values if 'prec' in daily_df_full else np.zeros(len(full_daily_range))
            rad_daily = daily_df_full['rad'].values if 'rad' in daily_df_full else np.zeros(len(full_daily_range))

            # Compute domain features on daily data FIRST
            domain_daily = {}
            if use_gdd:
                gdd_daily = _compute_gdd_series(result['tavg_raw'], gdd_base, gdd_upper)
                # For cumulative GDD: cumsum on daily, then SUM during aggregation
                # (not mean - you can't average cumulative values)
                domain_daily['cum_gdd_daily'] = np.cumsum(gdd_daily)

            if use_rue or use_farquhar:
                cum_prec_d = np.nancumsum(result['prec_raw'])   
                cum_rad_d = np.nancumsum(rad_daily)
                if use_rue:
                    domain_daily['rue'] = _compute_rue_series(
                        result['tavg_raw'], cum_prec_d, cum_rad_d, crop)
                if use_farquhar:
                    domain_daily['farquhar'] = _compute_farquhar_series(
                        result['tavg_raw'], cum_prec_d, cum_rad_d)

            # Add domain columns to daily df for resampling
            for k, v in domain_daily.items():
                daily_df_full[k] = v

            # Aggregate to target frequency
            # Use different aggregation methods for different feature types
            # - Weather features: mean (true temporal averaging)
            # - Cumulative GDD: max (take end-of-period cumulative value)
            # - RUE/Farquhar: mean (these are index-like, not cumulative)
            agg_methods = {col: 'mean' for col in weather_features_list}
            if use_gdd:
                agg_methods['cum_gdd_daily'] = 'max'  # was 'sum', use max for cumulative
            if use_rue:
                agg_methods['rue'] = 'mean'
            if use_farquhar:
                agg_methods['farquhar'] = 'mean'

            resampled = daily_df_full.resample(freq).agg(agg_methods)

            # Reindex to match exact target_dates
            expected_index = pd.date_range(start=target_dates[0], periods=seq_len, freq=freq)
            resampled = resampled.reindex(expected_index)

            # Build output with correct column order
            result['weather'] = np.full((seq_len, n_total), np.nan, dtype=np.float32)

            # Base weather features
            for j, col in enumerate(weather_features_list):
                if col in resampled.columns:
                    result['weather'][:, j] = resampled[col].values

            # Domain feature channels
            col_idx = n_weather
            if use_gdd and 'cum_gdd_daily' in resampled.columns:
                result['weather'][:, col_idx] = resampled['cum_gdd_daily'].values
                col_idx += 1
            if use_rue and 'rue' in resampled.columns:
                result['weather'][:, col_idx] = resampled['rue'].values
                col_idx += 1
            if use_farquhar and 'farquhar' in resampled.columns:
                result['weather'][:, col_idx] = resampled['farquhar'].values

            # Leave NaNs in place - normalization will impute to 0.0 (mean in z-score space)
    except Exception as e:
        logging.error(f"[{adm_id}] Weather extraction error: {e}")

    return result

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
                              debug: bool = False,
                              # New parameters for heat stress computation:
                              use_heat_stress_days: bool = False,
                              crop: str = 'maize',
                              tavg_series: Optional[np.ndarray] = None,
                              tmin_series: Optional[np.ndarray] = None,
                              tmax_series: Optional[np.ndarray] = None,
                              prec_series: Optional[np.ndarray] = None,
                              validity_mask: Optional[np.ndarray] = None,
                              ) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """
    Assemble the static feature vector for one location-year.

    Feature order (MUST match _get_static_feature_names() exactly):
      1. Soil properties
      2. Location properties  →  also extracts lat/lon as side-effect
      3. Crop calendar dates
      4. Explicit lat/lon      (conditional)
      5. Lagged yields         (conditional)
      6. Heat stress counts    (conditional)

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
        use_heat_stress_days: Whether to compute heat stress counts
        crop: Crop name for crop-specific thresholds
        tavg_series: Daily mean temperature array for heat stress computation
        tmin_series: Daily minimum temperature array for heat stress computation
        tmax_series: Daily maximum temperature array for heat stress computation
        prec_series: Daily precipitation array for heat stress computation
        validity_mask: Boolean mask of valid timesteps for heat stress counting

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
    # Renamed local variable to crop_calendar to avoid collision with crop parameter (string)
    if ("crop_season" in dataset._dfs_x and
            (adm_id, year) in dataset._dfs_x["crop_season"].index):
        crop_calendar = dataset._dfs_x["crop_season"].loc[(adm_id, year)]
    else:
        crop_calendar = pd.Series([np.nan] * 4, index=CROP_CALENDAR_DATES)

    for name, v in zip(CROP_CALENDAR_DATES, crop_calendar):
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

    # 6. Heat stress counts (static scalar features, computed from raw TS)
    if use_heat_stress_days:
        if (tavg_series is not None and tmin_series is not None and
                tmax_series is not None and prec_series is not None and
                validity_mask is not None):
            hs = _compute_heat_stress_counts(
                tavg_series, tmin_series, tmax_series, prec_series,
                crop, validity_mask
            )
            # Append in the EXACT order declared in _get_static_feature_names
            for key in ['heat_stress_days', 'frost_days', 'cold_stress_days',
                        'dry_days', 'wet_days', 'heat_stress_frac', 'dry_frac']:
                static_vals.append(hs[key])
        else:
            logging.warning(
                f"[{adm_id}] use_heat_stress_days=True but raw TS arrays not provided. "
                f"Appending NaN placeholders for heat stress features."
            )
            static_vals.extend([np.nan] * 7)

    return np.array(static_vals, dtype=np.float32), lat, lon

def _assemble_features(features: Dict, seq_len: int,
                       use_sota_features: bool,
                       weather_features_list: List[str],
                       use_gdd: bool = False,
                       use_rue: bool = False,
                       use_farquhar: bool = False) -> np.ndarray:
    """
    Concatenate time series feature arrays in consistent column order.

    Now accepts weather_features_list parameter instead of using global.

    Now includes domain features (GDD, RUE, Farquhar) when enabled.
    """
    n_weather = len(weather_features_list)
    n_domain = sum([use_gdd, use_rue, use_farquhar])
    n_weather_total = n_weather + n_domain  # weather array now includes domain channels
    n_rs = len(REMOTE_SENSING_FEATURES)      # 4
    n_sota = len(SOTA_TEMPORAL_VARS_LIST) if use_sota_features else 0

    X = np.zeros((seq_len, n_weather_total + n_rs + n_sota), dtype=np.float32)
    col = 0

    if 'weather' in features:
        # Assert exact shape match - _extract_weather_features now guarantees this
        # The weather array now includes base weather + domain features
        assert features['weather'].shape == (seq_len, n_weather_total), (
            f"Weather shape mismatch: expected ({seq_len}, {n_weather_total}), "
            f"got {features['weather'].shape}. This indicates a bug in "
            "_extract_weather_features - it should always return n_weather_total columns."
        )
        X[:, col:col + n_weather_total] = features['weather']
    col += n_weather_total

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
        # New domain feature flags:
        use_gdd: bool = False,
        use_heat_stress_days: bool = False,
        use_rue: bool = False,
        use_farquhar: bool = False,
        crop: str = 'maize',
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
        use_gdd: Add cumulative GDD as a time series channel
        use_heat_stress_days: Add heat stress day counts as static features
        use_rue: Add RUE index as a time series channel
        use_farquhar: Add Farquhar proxy as a time series channel
        crop: Crop name for crop-specific parameters

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

    # Crop-specific GDD thresholds from cybench.config
    # Base temp: 8°C maize (standard), 5°C wheat/other
    # Upper temp: 30°C maize, 25°C wheat/other (development stops above this)
    gdd_base = float(GDD_BASE_TEMP.get(crop, 8.0 if crop == 'maize' else 5.0))
    gdd_upper = float(GDD_UPPER_LIMIT.get(crop, 30.0 if crop == 'maize' else 25.0))

    logging.debug(f"Building sequence: {adm_id}, {year}, {aggregation}")

    # Crop season trimming
    crop_season_info = None
    if ("crop_season" in dataset._dfs_x and
            (adm_id, year) in dataset._dfs_x["crop_season"].index):
        crop_season_info = dataset._dfs_x["crop_season"].loc[(adm_id, year)]

    target_dates, seq_len, freq_str = _get_aggregation_params(aggregation, year, crop_season_info)

    features = {}

    # Extract weather features with optional domain features
    # Now returns a dict with weather array and raw daily arrays for heat stress
    weather_result = _extract_weather_features(
        dataset, adm_id, year, target_dates, aggregation, weather_features_list, debug,
        use_gdd=use_gdd, use_rue=use_rue, use_farquhar=use_farquhar,
        crop=crop, gdd_base=gdd_base, gdd_upper=gdd_upper,
    )
    features['weather'] = weather_result['weather']

    # Extract raw daily arrays for heat stress computation
    tavg_raw = weather_result.get('tavg_raw')
    tmin_raw = weather_result.get('tmin_raw')
    tmax_raw = weather_result.get('tmax_raw')
    prec_raw = weather_result.get('prec_raw')

    # For heat stress, we need a validity mask for the raw daily data
    # Create mask based on non-NaN values in the raw arrays
    if use_heat_stress_days and tavg_raw is not None:
        raw_validity_mask = ~np.isnan(tavg_raw)
    else:
        raw_validity_mask = None

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

    # Static features (now includes heat stress if enabled)
    features['static'], lat, lon = _extract_static_features(
        dataset, adm_id, year, include_spatial_features, None, None,
        lag_years,
        daily_df=None,  # No GDD growth stage features
        debug=debug,
        use_heat_stress_days=use_heat_stress_days,
        crop=crop,
        tavg_series=tavg_raw,
        tmin_series=tmin_raw,
        tmax_series=tmax_raw,
        prec_series=prec_raw,
        validity_mask=raw_validity_mask,
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

    # Pass weather_features_list and domain feature flags to _assemble_features
    X_ts = _assemble_features(features, seq_len, use_sota_features, weather_features_list,
                               use_gdd=use_gdd, use_rue=use_rue, use_farquhar=use_farquhar)
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

def _get_static_feature_names(
    include_spatial_features: bool,
    lag_years: int,
    use_heat_stress_days: bool = False,
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
      6. Heat stress counts (conditional)
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

    # Heat stress counts 
    if use_heat_stress_days:
        names.extend([
            'heat_stress_days',
            'frost_days',
            'cold_stress_days',
            'dry_days',
            'wet_days',
            'heat_stress_frac',
            'dry_frac',
        ]) 

    return names