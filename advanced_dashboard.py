from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from .advanced_monitoring import AdvancedHealthMonitor, create_monitoring_dashboard_data
from .advanced_ensemble import AdvancedEnsembleModel
from .data_pipeline import generate_synthetic_dataset, load_smart_csv
from .feature_engineering import create_feature_matrix
from .models import build_demo_models, ensemble_predict
from .utils import dataframe_to_csv_bytes


def create_advanced_dashboard():
    """Create advanced dashboard with comprehensive monitoring features."""
    st.set_page_config(
        page_title="Advanced Storage Health Monitor", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔧 Advanced Storage Health Monitor")
    st.caption("Comprehensive SMART analytics with advanced ensemble models and multi-dimensional health scoring")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Upload CSV", "Generate Synthetic", "Use Sample Data"],
        index=0
    )
    
    # Load data
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload SMART dataset (CSV)", 
            type="csv"
        )
        if uploaded_file is not None:
            telemetry = load_smart_csv(uploaded_file)
            if telemetry.empty:
                st.sidebar.warning("No valid data found. Using synthetic data.")
                telemetry = generate_synthetic_dataset(100)
        else:
            telemetry = generate_synthetic_dataset(50)
    elif data_source == "Generate Synthetic":
        samples = st.sidebar.slider("Sample Size", 50, 500, 200)
        telemetry = generate_synthetic_dataset(samples)
    else:
        telemetry = generate_synthetic_dataset(100)
    
    # Feature engineering
    features, metadata = create_feature_matrix(telemetry)
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    model_type = st.sidebar.selectbox(
        "Choose Model",
        ["Original Ensemble", "Advanced Ensemble", "Both Models"],
        index=0
    )
    
    # Build models
    if st.sidebar.button("🚀 Build Models"):
        with st.spinner("Building models..."):
            if model_type in ["Original Ensemble", "Both Models"]:
                st.session_state["original_models"] = build_demo_models(telemetry)
            
            if model_type in ["Advanced Ensemble", "Both Models"]:
                advanced_ensemble = AdvancedEnsembleModel()
                # Create synthetic labels for training
                labels = pd.Series(
                    pd.cut(
                        telemetry["smart_5"] + telemetry["smart_187"],
                        bins=3,
                        labels=["healthy", "warning", "fail"]
                    ).fillna("healthy"),
                    index=telemetry.index
                )
                advanced_ensemble.build_ensemble(features, labels)
                st.session_state["advanced_models"] = advanced_ensemble
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔍 Detailed Analysis", 
        "⚠️ Anomaly Detection", "📈 Predictions", "📋 Reports"
    ])
    
    with tab1:
        _create_overview_tab(features, metadata)
    
    with tab2:
        _create_detailed_analysis_tab(features, metadata)
    
    with tab3:
        _create_anomaly_detection_tab(features, metadata)
    
    with tab4:
        _create_predictions_tab(features, metadata)
    
    with tab5:
        _create_reports_tab(features, metadata)


def _create_overview_tab(features: pd.DataFrame, metadata: pd.DataFrame):
    """Create overview dashboard tab."""
    st.header("📊 Health Overview")
    
    # Get predictions based on available models
    if "original_models" in st.session_state:
        risk_df = ensemble_predict(
            st.session_state["original_models"], 
            features, 
            metadata
        )
        _display_original_overview(risk_df, features)
    
    if "advanced_models" in st.session_state:
        advanced_risk_df = st.session_state["advanced_models"].predict_ensemble_risk(
            features, metadata
        )
        _display_advanced_overview(advanced_risk_df, features)


def _display_original_overview(risk_df: pd.DataFrame, features: pd.DataFrame):
    """Display original model overview."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Drives", len(risk_df))
    
    with col2:
        avg_risk = risk_df["risk_score"].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.3f}")
    
    with col3:
        critical_drives = len(risk_df[risk_df["status"] == "FAIL"])
        st.metric("Critical Drives", critical_drives, delta=None)
    
    with col4:
        healthy_drives = len(risk_df[risk_df["status"] == "HEALTHY"])
        st.metric("Healthy Drives", healthy_drives, delta=None)
    
    # Risk distribution chart
    st.subheader("Risk Distribution")
    fig = px.pie(
        risk_df, 
        names="status", 
        title="Drive Status Distribution",
        color_discrete_map={
            "FAIL": "firebrick", 
            "WARN": "gold", 
            "MONITOR": "darkorange", 
            "HEALTHY": "seagreen"
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk projection chart
    st.subheader("Risk Projection")
    fig = px.bar(
        risk_df.sort_values("risk_score", ascending=False).head(20),
        x="drive_id", 
        y="risk_score",
        color="status",
        color_discrete_map={
            "FAIL": "firebrick", 
            "WARN": "gold", 
            "MONITOR": "darkorange", 
            "HEALTHY": "seagreen"
        },
        title="Top 20 Drives by Risk Score"
    )
    st.plotly_chart(fig, use_container_width=True)


def _display_advanced_overview(risk_df: pd.DataFrame, features: pd.DataFrame):
    """Display advanced model overview."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Drives", len(risk_df))
    
    with col2:
        avg_health = risk_df["comprehensive_health_score"].mean()
        st.metric("Avg Health Score", f"{avg_health:.3f}")
    
    with col3:
        critical_drives = len(risk_df[risk_df["advanced_risk_category"] == "CRITICAL"])
        st.metric("Critical Drives", critical_drives, delta=None)
    
    with col4:
        avg_confidence = risk_df["ensemble_confidence"].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    # Health distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Health Category Distribution")
        fig = px.pie(
            risk_df, 
            names="health_category", 
            title="Health Categories"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Category Distribution")
        fig = px.pie(
            risk_df, 
            names="advanced_risk_category", 
            title="Risk Categories"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced risk projection
    st.subheader("Advanced Risk Analysis")
    fig = px.scatter(
        risk_df,
        x="comprehensive_health_score",
        y="failure_probability",
        color="advanced_risk_category",
        size="ensemble_confidence",
        hover_data=["drive_id"],
        title="Health Score vs Failure Probability"
    )
    st.plotly_chart(fig, use_container_width=True)


def _create_detailed_analysis_tab(features: pd.DataFrame, metadata: pd.DataFrame):
    """Create detailed analysis tab."""
    st.header("🔍 Detailed Feature Analysis")
    
    # Feature selection
    selected_features = st.multiselect(
        "Select Features to Analyze",
        options=features.columns.tolist(),
        default=features.columns[:10].tolist()
    )
    
    if selected_features:
        # Feature distribution
        st.subheader("Feature Distributions")
        for feature in selected_features[:6]:  # Limit to 6 features
            fig = px.histogram(
                features[feature], 
                title=f"Distribution of {feature}",
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Feature Correlations")
        corr_matrix = features[selected_features].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis
        if "recorded_at" in metadata.columns:
            st.subheader("Time Series Analysis")
            for feature in selected_features[:3]:  # Limit to 3 features
                fig = px.line(
                    x=metadata["recorded_at"],
                    y=features[feature],
                    title=f"Time Series: {feature}"
                )
                st.plotly_chart(fig, use_container_width=True)


def _create_anomaly_detection_tab(features: pd.DataFrame, metadata: pd.DataFrame):
    """Create anomaly detection tab."""
    st.header("⚠️ Anomaly Detection")
    
    monitor = AdvancedHealthMonitor()
    
    # Anomaly detection method
    method = st.selectbox(
        "Detection Method",
        ["isolation_forest", "statistical", "zscore"],
        index=0
    )
    
    # Detect anomalies
    anomalies = monitor.detect_anomalies(features, method)
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Drives", len(features))
    
    with col2:
        anomalous_count = anomalies.sum()
        st.metric("Anomalous Drives", anomalous_count)
    
    with col3:
        anomaly_rate = anomalous_count / len(features) * 100
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    
    # Anomaly visualization
    st.subheader("Anomaly Analysis")
    
    # Create 2D visualization using PCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.select_dtypes(include=[float, int]))
    
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(scaled_features)
    
    fig = px.scatter(
        x=features_2d[:, 0],
        y=features_2d[:, 1],
        color=anomalies,
        title="Anomaly Detection (PCA Visualization)",
        color_discrete_map={True: "red", False: "blue"},
        labels={"color": "Anomaly", "x": "PC1", "y": "PC2"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomalous drives details
    if anomalies.any():
        st.subheader("Anomalous Drives Details")
        anomalous_drives = metadata[anomalies]
        st.dataframe(anomalous_drives, use_container_width=True)


def _create_predictions_tab(features: pd.DataFrame, metadata: pd.DataFrame):
    """Create predictions tab."""
    st.header("📈 Failure Predictions")
    
    monitor = AdvancedHealthMonitor()
    
    # Prediction horizon
    days_ahead = st.slider("Prediction Horizon (Days)", 7, 90, 30)
    
    # Generate predictions
    failure_predictions = monitor.predict_failure_time(features, days_ahead)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_risk_count = (failure_predictions["failure_probability"] > 0.7).sum()
        st.metric("High Risk Drives", high_risk_count)
    
    with col2:
        avg_rul = failure_predictions["predicted_rul_days"].mean()
        st.metric("Avg Predicted RUL", f"{avg_rul:.1f} days")
    
    with col3:
        critical_rul = (failure_predictions["predicted_rul_days"] < 7).sum()
        st.metric("Critical RUL (<7 days)", critical_rul)
    
    with col4:
        avg_failure_prob = failure_predictions["failure_probability"].mean()
        st.metric("Avg Failure Probability", f"{avg_failure_prob:.3f}")
    
    # Predictions visualization
    st.subheader("Failure Probability Distribution")
    fig = px.histogram(
        failure_predictions["failure_probability"],
        title="Failure Probability Distribution",
        nbins=20,
        labels={"value": "Failure Probability", "count": "Number of Drives"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # RUL predictions
    st.subheader("Remaining Useful Life Predictions")
    fig = px.bar(
        failure_predictions.sort_values("predicted_rul_days").head(20),
        x="drive_id",
        y="predicted_rul_days",
        title="RUL Predictions (Top 20 Most Critical)",
        labels={"predicted_rul_days": "Predicted RUL (Days)"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed predictions table
    st.subheader("Detailed Predictions")
    prediction_details = metadata.copy()
    prediction_details["failure_probability"] = failure_predictions["failure_probability"]
    prediction_details["predicted_rul_days"] = failure_predictions["predicted_rul_days"]
    
    st.dataframe(
        prediction_details[["drive_id", "failure_probability", "predicted_rul_days"]],
        use_container_width=True
    )


def _create_reports_tab(features: pd.DataFrame, metadata: pd.DataFrame):
    """Create reports tab."""
    st.header("📋 Health Reports")
    
    monitor = AdvancedHealthMonitor()
    
    # Generate comprehensive report
    if st.button("🔄 Generate Health Report"):
        with st.spinner("Generating comprehensive health report..."):
            report = monitor.generate_health_report(features, metadata)
            
            # Display summary
            st.subheader("📊 Executive Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Drives", report["summary"]["total_drives"])
                st.metric("Healthy Drives", report["summary"]["healthy_drives"])
            
            with col2:
                st.metric("Drives at Risk", report["summary"]["drives_at_risk"])
                st.metric("Avg Health Score", f"{report['summary']['avg_health_score']:.3f}")
            
            with col3:
                st.metric("Anomalous Drives", report["summary"]["anomalous_drives"])
            
            # Health distribution
            st.subheader("Health Distribution")
            health_dist = report["health_distribution"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=list(health_dist.values()),
                    names=list(health_dist.keys()),
                    title="Health Categories"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create bar chart
                fig = px.bar(
                    x=list(health_dist.keys()),
                    y=list(health_dist.values()),
                    title="Health Category Counts"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Critical drives
            if report["critical_drives"]:
                st.subheader("🚨 Critical Drives")
                critical_df = pd.DataFrame({"drive_id": report["critical_drives"]})
                st.dataframe(critical_df, use_container_width=True)
            
            # High risk predictions
            if report["high_risk_predictions"]:
                st.subheader("⚠️ High Risk Drives")
                risk_df = pd.DataFrame({"drive_id": report["high_risk_predictions"]})
                st.dataframe(risk_df, use_container_width=True)
            
            # Download report
            st.subheader("📥 Download Report")
            
            # Create report data
            report_data = {
                "summary": report["summary"],
                "health_distribution": report["health_distribution"],
                "critical_drives": report["critical_drives"],
                "anomalous_drives": report["anomalous_drives"],
                "high_risk_predictions": report["high_risk_predictions"]
            }
            
            # Convert to downloadable format
            report_json = pd.Series(report_data).to_json(indent=2)
            st.download_button(
                label="Download Health Report (JSON)",
                data=report_json,
                file_name="storage_health_report.json",
                mime="application/json"
            )


if __name__ == "__main__":
    create_advanced_dashboard()
