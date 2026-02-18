from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURE_COLUMNS = [
    "smart_5",
    "smart_187",
    "smart_197", 
    "smart_198",
    "smart_188",
    "smart_171",
    "smart_172",
    "smart_177",
    "smart_182",
    "smart_194",
    "smart_9",
    "smart_12",
    "smart_241",
    # Additional critical SMART attributes
    "smart_1",    # Read Error Rate
    "smart_3",    # Spin-Up Time
    "smart_4",    # Start/Stop Count
    "smart_7",    # Seek Error Rate
    "smart_10",   # Spin Retry Count
    "smart_193",  # Load/Unload Cycle Count
    "smart_199",  # UltraDMA CRC Error Count
    "smart_240",  # Head Flying Hours
]

ROLLING_COLUMNS = [
    "smart_5",
    "smart_187",
    "smart_171",
    "smart_172",
    "smart_177",
    "smart_182",
    "smart_194",
    # Additional critical attributes for rolling analysis
    "smart_1",
    "smart_3", 
    "smart_7",
    "smart_10",
    "smart_193",
    "smart_199",
]

DERIVED_FEATURES = [
    "delta_realloc",
    "delta_program_fail", 
    "rolling_std_realloc",
    "error_rate",
    "wear_rate",
    "thermal_velocity",
    "spare_pressure",
    "delta_error",
    "error_acceleration",
    "temperature_variance",
    # New advanced features
    "critical_error_ratio",
    "mechanical_stress_index", 
    "electrical_stress_index",
    "aging_factor",
    "performance_degradation",
    "health_score_trend",
    "volatility_index",
    "cumulative_wear",
    "thermal_stress_cycles",
    "error_burst_indicator",
    "reliability_degradation",
]


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill missing columns with zeros and coerce to float."""
    for column in columns:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return df


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling/delta-derived diagnostic signals with advanced failure prediction features."""
    df = df.copy()
    
    # Original features
    df["delta_realloc"] = df["smart_5"].diff().fillna(0.0)
    df["delta_program_fail"] = df["smart_171"].diff().fillna(0.0)
    df["rolling_std_realloc"] = df["smart_5"].rolling(window=7, min_periods=1).std().fillna(0.0)
    df["error_rate"] = (df["smart_187"] + df["smart_171"] + df["smart_172"]) / (1 + df["smart_9"])
    df["wear_rate"] = df["smart_177"] / (1 + df["smart_12"])
    df["thermal_velocity"] = df["smart_194"] - df["smart_194"].rolling(window=7, min_periods=1).mean().fillna(0.0)
    df["spare_pressure"] = np.clip(1 - df["smart_182"] / 100, 0, 1)
    df["delta_error"] = (df["smart_187"] + df["smart_171"] + df["smart_172"]).diff().fillna(0.0)
    df["error_acceleration"] = df["delta_error"].diff().fillna(0.0)
    df["temperature_variance"] = df["smart_194"].rolling(window=7, min_periods=1).var().fillna(0.0)
    
    # New advanced features
    # Critical error ratio - proportion of severe errors
    critical_errors = df["smart_5"] + df["smart_187"] + df["smart_198"]
    total_errors = df["smart_1"] + df["smart_5"] + df["smart_187"] + df["smart_197"] + df["smart_198"]
    df["critical_error_ratio"] = np.where(total_errors > 0, critical_errors / total_errors, 0.0)
    
    # Mechanical stress index - combines mechanical wear indicators
    mechanical_factors = [
        df["smart_3"].fillna(0),    # Spin-Up Time
        df["smart_4"].fillna(0),    # Start/Stop Count  
        df["smart_7"].fillna(0),    # Seek Error Rate
        df["smart_10"].fillna(0),   # Spin Retry Count
        df["smart_193"].fillna(0),  # Load/Unload Cycle Count
    ]
    df["mechanical_stress_index"] = np.mean([np.clip(f / (f.max() + 1e-6), 0, 1) for f in mechanical_factors], axis=0)
    
    # Electrical stress index - electrical component stress
    electrical_factors = [
        df["smart_1"].fillna(0),    # Read Error Rate
        df["smart_199"].fillna(0),  # CRC Error Count
        df["smart_240"].fillna(0),  # Head Flying Hours
    ]
    df["electrical_stress_index"] = np.mean([np.clip(f / (f.max() + 1e-6), 0, 1) for f in electrical_factors], axis=0)
    
    # Aging factor - combination of age and usage
    max_hours = df["smart_9"].max() if df["smart_9"].max() > 0 else 1
    max_cycles = df["smart_12"].max() if df["smart_12"].max() > 0 else 1
    df["aging_factor"] = (df["smart_9"] / max_hours * 0.7 + df["smart_12"] / max_cycles * 0.3)
    
    # Performance degradation - trend analysis of performance metrics
    df["performance_degradation"] = (
        df["smart_7"].rolling(window=14).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False).fillna(0)
    )
    
    # Health score trend - overall health trajectory
    health_components = [
        -df["smart_5"] / (df["smart_5"].max() + 1),      # Lower is better
        -df["smart_187"] / (df["smart_187"].max() + 1),  # Lower is better
        -df["error_rate"] / (df["error_rate"].max() + 1),   # Lower is better
        df["smart_182"] / 100,                           # Higher is better
    ]
    health_score = np.mean(health_components, axis=0)
    df["health_score_trend"] = health_score.rolling(window=14).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False).fillna(0)
    
    # Volatility index - measure of instability in SMART attributes
    volatility_attrs = ["smart_5", "smart_187", "smart_194", "smart_171"]
    volatilities = [df[attr].rolling(window=7).std().fillna(0) for attr in volatility_attrs]
    df["volatility_index"] = np.mean(volatilities, axis=0)
    
    # Cumulative wear - accumulated damage over time
    df["cumulative_wear"] = (df["smart_5"] + df["smart_187"] + df["smart_197"] + df["smart_198"]).cumsum()
    
    # Thermal stress cycles - temperature fluctuation stress
    temp_mean = df["smart_194"].rolling(window=24).mean().fillna(df["smart_194"].iloc[0])
    temp_deviation = np.abs(df["smart_194"] - temp_mean)
    df["thermal_stress_cycles"] = (temp_deviation > 5).astype(int).rolling(window=24).sum().fillna(0)
    
    # Error burst indicator - sudden spikes in errors
    error_threshold = df["error_rate"].rolling(window=168).quantile(0.95).fillna(0)  # 7-day threshold
    df["error_burst_indicator"] = (df["error_rate"] > error_threshold).astype(int).rolling(window=6).sum().fillna(0)
    
    # Reliability degradation - long-term trend in reliability
    reliability_score = 1 - np.clip(df["error_rate"], 0, 1)
    df["reliability_degradation"] = -reliability_score.rolling(window=30).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=False).fillna(0)
    
    # Rolling statistics for all columns
    for window in (7, 30):
        for column in ROLLING_COLUMNS:
            df[f"{column}_{window}d_mean"] = df[column].rolling(window=window, min_periods=1).mean().fillna(0.0)
            df[f"{column}_{window}d_std"] = df[column].rolling(window=window, min_periods=1).std().fillna(0.0)
            df[f"{column}_{window}d_max"] = df[column].rolling(window=window, min_periods=1).max().fillna(0.0)
            df[f"{column}_{window}d_min"] = df[column].rolling(window=window, min_periods=1).min().fillna(0.0)
    
    return df


def create_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce the normalized numeric matrix plus metadata for each drive."""
    df = df.copy()
    df = ensure_numeric(df, BASE_FEATURE_COLUMNS + ["smart_194"])
    df = compute_derived_metrics(df)
    feature_columns = BASE_FEATURE_COLUMNS + DERIVED_FEATURES
    features = df[feature_columns].astype(float).fillna(0.0)
    for column in ("drive_id", "vendor", "device_type", "label"):
        if column not in df.columns:
            default = "unknown" if column != "drive_id" else pd.RangeIndex(start=0, stop=len(df)).astype(str)
            df[column] = default
    metadata = df[["drive_id", "vendor", "device_type", "label"]].copy()
    return features, metadata
