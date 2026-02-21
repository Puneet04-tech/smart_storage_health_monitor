from __future__ import annotations

import pandas as pd
from typing import Any
import plotly.express as px
import streamlit as st

from data_pipeline import generate_synthetic_dataset, load_smart_csv
from feature_engineering import create_feature_matrix
from monitoring import forecast_temperature_trend
from models import build_demo_models, ensemble_predict
from utils import dataframe_to_csv_bytes

st.set_page_config(page_title="Smart Storage Health Monitor", layout="wide")


def _build_projection_chart(risk_df: pd.DataFrame) -> None:
    fig = px.bar(
        risk_df,
        x="drive_id",
        y="risk_score",
        color="status",
        color_discrete_map={"FAIL": "firebrick", "WARN": "gold", "MONITOR": "darkorange", "HEALTHY": "seagreen"},
        title="Drive Risk Overview",
    )
    st.plotly_chart(fig, use_container_width=True)


def _status_metrics(risk_df: pd.DataFrame) -> None:
    counts = risk_df["status"].value_counts().reindex(["FAIL", "WARN", "MONITOR", "HEALTHY"], fill_value=0)
    cols = st.columns(4)
    for idx, status in enumerate(counts.index):
        with cols[idx]:
            st.metric(label=f"{status} drives", value=int(counts[status]), delta=None)


def _device_type_summary(risk_df: pd.DataFrame, metadata: pd.DataFrame, telemetry: pd.DataFrame) -> dict[str, dict[str, Any]]:
    smart_columns = ["smart_5", "smart_171", "smart_172", "smart_187", "smart_177", "smart_182"]
    summary: dict[str, dict[str, Any]] = {}
    for dtype in risk_df["device_type"].dropna().unique():
        subset = risk_df[risk_df["device_type"] == dtype]
        if subset.empty:
            continue
        drive_ids = subset["drive_id"].unique()
        tele_subset = telemetry.copy()
        if "drive_id" in tele_subset.columns:
            tele_subset = tele_subset[tele_subset["drive_id"].isin(drive_ids)]
        else:
            tele_subset = pd.DataFrame()
        smart_means = (tele_subset[smart_columns].mean().round(2).to_dict() if not tele_subset.empty else {})
        status_counts = subset["status"].value_counts().to_dict()
        summary[dtype] = {
            "count": len(subset),
            "avg_risk": float(subset["risk_score"].mean()),
            "status_counts": status_counts,
            "smart_means": smart_means,
        }
    return summary


def main() -> None:
    st.title("Smart Storage Health Monitor")
    st.caption("Multimodal SMART analytics with XGBoost, Random Forest, LSTM, Isolation Forest, and adaptive risk scoring.")

    uploaded = st.file_uploader("Upload SMART dataset (CSV)", type="csv")
    if uploaded is not None:
        telemetry = load_smart_csv(uploaded)
        if telemetry.empty:
            st.warning("Uploaded file contained no rows; falling back to synthetic sample data.")
            telemetry = generate_synthetic_dataset(20)
        else:
            st.success("Loaded SMART telemetry from your file.")
    else:
        st.info("Using a synthetic sample dataset. Upload your SMART CSV to replace it.")
        telemetry = generate_synthetic_dataset(20)

    features, metadata = create_feature_matrix(telemetry)
    numeric_feature_count = features.select_dtypes(include="number").shape[1]

    needs_model_build = "smart_models" not in st.session_state
    if not needs_model_build:
        scaler = st.session_state["smart_models"].get("scaler")
        stored_feature_count = getattr(scaler, "n_features_in_", None)
        if stored_feature_count != numeric_feature_count:
            needs_model_build = True
    if needs_model_build:
        st.session_state["smart_models"] = build_demo_models(telemetry)

    risk_df = ensemble_predict(st.session_state["smart_models"], features, metadata)
    average_risk = risk_df["risk_score"].mean() if not risk_df.empty else 0.0
    st.metric("Average risk score", f"{average_risk:.2f}", delta=None)
    st.caption("Per-drive status may differ because of vendor-specific modeling and anomaly boosts.")

    device_summary = _device_type_summary(risk_df, metadata, telemetry)
    if device_summary:
        st.subheader("Device-type health summary")
        columns = st.columns(len(device_summary))
        for col, (dtype, info) in zip(columns, device_summary.items()):
            col.markdown(f"**{dtype} drives ({info['count']})**")
            col.metric("Avg. risk", f"{info['avg_risk']:.2f}")
            status = info["status_counts"]
            col.write(
                " | ".join(f"{key}: {status.get(key, 0)}" for key in ["FAIL", "WARN", "MONITOR", "HEALTHY"])
            )
            if info["smart_means"]:
                stats_df = pd.DataFrame.from_dict(info["smart_means"], orient="index", columns=["mean"]).round(2)
                col.table(stats_df)

    _status_metrics(risk_df)
    _build_projection_chart(risk_df)

    with st.expander("Drive details"):
        st.dataframe(risk_df["drive_id status risk_score xgb_fail_prob rf_fail_prob rul_days anomaly_score".split()], use_container_width=True)
        st.download_button(
            label="Download risk report",
            data=dataframe_to_csv_bytes(risk_df),
            file_name="smart_storage_risk_report.csv",
            mime="text/csv",
        )

    st.header("Temperature forecast vs historic SMART 194")
    forecast = forecast_temperature_trend(telemetry)
    if not forecast.empty:
        fig = px.line(forecast, x="ds", y="yhat", title="Prophet temperature forecast")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough historical temperature rows to generate a forecast.")


if __name__ == "__main__":
    main()
