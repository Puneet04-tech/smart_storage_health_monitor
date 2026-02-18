from __future__ import annotations

from typing import Tuple

import pandas as pd
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
    if not PROPHET_AVAILABLE:
        return pd.DataFrame()
    
    if "recorded_at" not in df or "smart_194" not in df:
        return pd.DataFrame()
    series = (
        df[["recorded_at", "smart_194"]]
        .rename(columns={"recorded_at": "ds", "smart_194": "y"})
        .reset_index(drop=True)
    )
    model = Prophet()
    model.fit(series)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
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
