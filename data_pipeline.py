from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd

SMART_COLUMNS = [
    "smart_5",
    "smart_187", 
    "smart_197",
    "smart_198",
    "smart_188",
    "smart_194",
    "smart_9",
    "smart_12",
    "smart_171",
    "smart_172",
    "smart_177",
    "smart_182",
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

SSD_MODELS = ["Samsung", "Micron", "Kingston", "Western Digital" ]
HDD_MODELS = ["Seagate", "Western Digital", "Toshiba", "HGST"]

LABEL_MAP = {"healthy": 0, "warning": 1, "fail": 2}


def load_smart_csv(source: Union[str, Path, BytesIO, Iterable[str]]) -> pd.DataFrame:
    """Load SMART telemetry from disk, buffer, or iterator of lines."""
    if isinstance(source, BytesIO):
        source.seek(0)
        return pd.read_csv(source)

    if hasattr(source, "read"):
        source.seek(0)
        return pd.read_csv(source)  # file-like object

    if isinstance(source, Iterable) and not isinstance(source, (str, Path)):
        return pd.read_csv(StringIO("\n".join(source)))

    path = Path(source)
    return pd.read_csv(path)


def detect_device_type(df: pd.DataFrame) -> pd.Series:
    """Tag each row as HDD or SSD using the vendor/model hints."""
    vendor_col = pd.Series("")
    for candidate in ("model", "vendor", "model_name"):
        if candidate in df.columns and df[candidate].dtype == object:
            vendor_col = df[candidate].astype(str)
            break
    vendor_col = vendor_col.str.lower().fillna("")
    is_ssd = vendor_col.str.contains("ssd|nvme|samsung|micron")
    return pd.Series(np.where(is_ssd, "SSD", "HDD"), index=df.index)


def clean_smart_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns, ensure SMART attributes exist, and enforce types."""
    df = df.copy()
    df["recorded_at"] = pd.to_datetime(df.get("recorded_at", pd.Timestamp.now()))
    df["vendor"] = df.get("vendor", "Unknown").astype(str)
    df["drive_id"] = df.get("drive_id", df.index.astype(str))

    for column in SMART_COLUMNS:
        df[column] = pd.to_numeric(df.get(column, 0), errors="coerce").fillna(0).astype(float)

    df["device_type"] = detect_device_type(df)
    df["rul_days"] = pd.to_numeric(df.get("rul_days", np.nan), errors="coerce").fillna(30)
    df["label"] = df.get("label", "healthy").astype(str)
    return df


def generate_synthetic_dataset(
    samples: int = 400, *, seed: int = 42
) -> pd.DataFrame:
    """Create a synthetic Backblaze-style dataset for HDD and SSD drives."""
    rng = np.random.default_rng(seed)
    vendor_choices = HDD_MODELS + SSD_MODELS
    device_types = rng.choice(["HDD", "SSD"], size=samples, p=[0.6, 0.4])
    vendor = rng.choice(vendor_choices, size=samples)

    smart = {
        # Original SMART attributes
        "smart_5": rng.integers(0, 80, size=samples),
        "smart_187": rng.integers(0, 120, size=samples),
        "smart_197": rng.integers(0, 40, size=samples),
        "smart_198": rng.integers(0, 10, size=samples),
        "smart_188": rng.integers(0, 50, size=samples),
        "smart_194": rng.uniform(30, 70, size=samples),
        "smart_9": rng.integers(1000, 20000, size=samples),
        "smart_12": rng.integers(100, 1000, size=samples),
        "smart_171": rng.integers(0, 80, size=samples),
        "smart_172": rng.integers(0, 60, size=samples),
        "smart_177": rng.random(size=samples) * 100,
        "smart_182": rng.uniform(10, 100, size=samples),
        "smart_241": rng.integers(0, 200, size=samples),
        
        # New SMART attributes
        "smart_1": rng.integers(0, 100, size=samples),      # Read Error Rate
        "smart_3": rng.uniform(1000, 5000, size=samples),  # Spin-Up Time
        "smart_4": rng.integers(100, 5000, size=samples),   # Start/Stop Count
        "smart_7": rng.integers(0, 50, size=samples),       # Seek Error Rate
        "smart_10": rng.integers(0, 30, size=samples),      # Spin Retry Count
        "smart_193": rng.integers(1000, 100000, size=samples), # Load/Unload Cycle Count
        "smart_199": rng.integers(0, 100, size=samples),    # CRC Error Count
        "smart_240": rng.integers(0, 10000, size=samples),   # Head Flying Hours
    }

    df = pd.DataFrame(
        {
            "drive_id": [f"drive_{i}" for i in range(samples)],
            "vendor": vendor,
            "device_type": device_types,
            "recorded_at": pd.date_range(end=pd.Timestamp.now(), periods=samples, freq="H"),
            **smart,
        }
    )

    # Enhanced scoring algorithm with new attributes
    score = (
        df["smart_5"] / 80 +
        df["smart_187"] / 120 +
        df["smart_171"] / 80 +
        df["smart_172"] / 60 +
        df["smart_1"] / 100 +          # Read errors
        df["smart_7"] / 50 +           # Seek errors
        df["smart_10"] / 30 +          # Spin retries
        df["smart_199"] / 100 +         # CRC errors
        (70 - df["smart_182"]) / 70 +  # Lower spare is bad
        np.abs(df["smart_194"] - 45) / 25  # Temperature deviation from optimal
    )
    
    df["score"] = score / 9  # Normalize by number of factors
    df["label"] = pd.cut(df["score"], bins=[-1, 0.35, 0.65, 1.1], labels=["healthy", "warning", "fail"]).astype(str)
    df["rul_days"] = np.where(
        df["label"] == "fail", 
        rng.integers(1, 10, size=samples), 
        np.where(
            df["label"] == "warning",
            rng.integers(10, 30, size=samples),
            rng.integers(30, 120, size=samples)
        )
    )
    return clean_smart_dataframe(df)
