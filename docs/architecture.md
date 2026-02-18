# Smart Storage Health Monitor Architecture

- **Data sources:** SMART telemetry from HDDs/SSDs via smartctl, CrystalDiskInfo, or vendor exports. The ingestion layer detects device type and normalizes SMART IDs.
- **Feature engineering:** Vendor-specific branches compute deltas, rolling statistics, temperature velocities, wear rates, and spare-pressure metrics; shared features unify HDD/SSD insights before modeling.
- **ML ensemble:** Trained XGBoost multi-class classifier, Random Forest consensus, LSTM RUL regression, Isolation Forest anomaly detection, and Prophet temperature forecasting feed a weighted risk aggregation layer.
- **Risk scoring:** Weighted blend (0.6 XGBoost + 0.25 Random Forest + 0.15 inverse RUL) with anomaly acceleration boosts yields FAIL/WARN/MONITOR/HEALTHY decisions.
- **Monitoring & explainability:** KS test drift detection, SHAP explanations, and Prophet thermal forecasts ensure live validation.
- **Deployment:** Streamlit dashboard + SQLite logging for Hackathon-ready demos; optional Hugging Face Spaces or FastAPI + Kubernetes for production scale.
