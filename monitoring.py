from __future__ import annotations

from typing import Tuple

import pandas as pd
import numpy as np
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from scipy.stats import ks_2samp


def detect_distribution_drift(
    reference: pd.Series, current: pd.Series
) -> tuple[float, bool]:
    """Return KS p-value and whether a drift threshold is exceeded."""
    if reference.empty or current.empty:
        return 1.0, False
    stat, p_value = ks_2samp(reference, current)
    return p_value, p_value < 0.05


def forecast_temperature_trend(df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    """Return Prophet forecast of temperatures (ds/y columns expected)."""
    # If Prophet available, use it; otherwise attempt a simple linear fallback
    if "recorded_at" not in df or "smart_194" not in df:
        return pd.DataFrame()

    series = (
        df[["recorded_at", "smart_194"]]
        .rename(columns={"recorded_at": "ds", "smart_194": "y"})
        .reset_index(drop=True)
    )

    # Ensure datetimes are parsed correctly (CSV may contain strings)
    series["ds"] = pd.to_datetime(series["ds"], errors="coerce")
    # drop rows missing datetime or target
    series = series.dropna(subset=["ds", "y"]).reset_index(drop=True)
    if series.empty:
        return pd.DataFrame()

    if PROPHET_AVAILABLE:
        model = Prophet()
        model.fit(series)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    # Fallback: simple linear regression on available points if enough rows
    if series.shape[0] < 5:
        return pd.DataFrame()

    # convert datetimes to ordinal floats (days) for regression
    x = series["ds"].astype("datetime64[ns]").astype("int64") / 1e9 / 86400.0
    y = series["y"].astype(float)
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    last_ts = series["ds"].max()
    # generate future ds values using median cadence of input (fallback to 1h)
    diffs = series["ds"].diff().dropna()
    freq = diffs.median() if not diffs.empty else pd.Timedelta(hours=1)
    future_ds = pd.date_range(start=last_ts + freq, periods=periods, freq=freq)
    future_x = future_ds.astype("datetime64[ns]").astype("int64") / 1e9 / 86400.0
    yhat = intercept + slope * future_x

    # create forecast frame matching Prophet-like columns
    forecast = pd.DataFrame({"ds": future_ds, "yhat": yhat})
    resid = y - (intercept + slope * x)
    sigma = float(np.std(resid)) if len(resid) > 1 else 0.0
    forecast["yhat_lower"] = forecast["yhat"] - 1.96 * sigma
    forecast["yhat_upper"] = forecast["yhat"] + 1.96 * sigma
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def explain_model_with_shap(model: object, features: pd.DataFrame):
    """Generate SHAP values for a set of features if possible."""
    if not SHAP_AVAILABLE or features.empty:
        return None
    try:
        explainer = shap.Explainer(model)
        return explainer(features)
    except Exception:
        return None
