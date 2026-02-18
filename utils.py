from __future__ import annotations

import pandas as pd


STATUS_THRESHOLDS = [0.8, 0.5, 0.2]
STATUS_LABELS = ["FAIL", "WARN", "MONITOR", "HEALTHY"]
STATUS_COLORS = {"FAIL": "danger", "WARN": "warning", "MONITOR": "info", "HEALTHY": "success"}


def risk_status(score: float) -> str:
    """Return a human-readable risk bucket from a blended score."""
    if score >= STATUS_THRESHOLDS[0]:
        return STATUS_LABELS[0]
    if score >= STATUS_THRESHOLDS[1]:
        return STATUS_LABELS[1]
    if score >= STATUS_THRESHOLDS[2]:
        return STATUS_LABELS[2]
    return STATUS_LABELS[3]


def risk_color(status: str) -> str:
    return STATUS_COLORS.get(status, "secondary")


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")
