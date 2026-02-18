# 🖥️ Smart Storage Health Monitor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning system for predicting disk drive failures using SMART (Self-Monitoring, Analysis, and Reporting Technology) data with advanced ensemble models and real-time monitoring capabilities.

## ✨ Key Features

### 🤖 Advanced Machine Learning
- **7 ML Algorithms**: XGBoost, Random Forest, LSTM, SVM, MLP, Naive Bayes, AdaBoost
- **Ensemble Methods**: Weighted voting and soft voting ensembles
- **Hyperparameter Optimization**: Optuna-based tuning for optimal performance
- **Feature Selection**: Multi-method feature importance analysis

### 📊 Comprehensive Monitoring
- **22 SMART Attributes**: Extended monitoring beyond basic attributes
- **25+ Derived Features**: Advanced feature engineering for better predictions
- **Multi-dimensional Health Scoring**: Error, Performance, Thermal, Mechanical health
- **Real-time Anomaly Detection**: Isolation Forest, Statistical, Z-score methods

### 📈 Predictive Analytics
- **Failure Prediction**: Multi-algorithm consensus with confidence scoring
- **RUL Estimation**: LSTM-based Remaining Useful Life prediction
- **Trend Analysis**: Linear and non-linear trend detection
- **Early Warning System**: Multi-level alerting system

### 🎯 Interactive Dashboard
- **5 Comprehensive Tabs**: Overview, Analysis, Anomalies, Predictions, Reports
- **Real-time Visualizations**: Interactive charts and graphs
- **Executive Reports**: Automated health reports and insights
- **Data Upload Support**: CSV upload and synthetic data generation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-storage-health-monitor.git
cd smart-storage-health-monitor

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Basic Usage

1. **Upload Data**: Upload your SMART data CSV file or generate synthetic data
2. **Build Models**: Train the ensemble models on your data
3. **Monitor Health**: View comprehensive health scores and risk assessments
4. **Analyze Trends**: Examine feature distributions and correlations
5. **Detect Anomalies**: Identify unusual drive behavior patterns
6. **Predict Failures**: Get RUL estimates and failure probabilities

## 📋 Requirements

### Core Dependencies
- `pandas >= 1.5.0` - Data manipulation
- `numpy >= 1.24.0` - Numerical computations
- `scikit-learn >= 1.3.0` - Machine learning algorithms
- `xgboost >= 1.7.0` - Gradient boosting
- `streamlit >= 1.28.0` - Web dashboard
- `plotly >= 5.15.0` - Interactive visualizations

### Optional Dependencies
- `tensorflow >= 2.12.0` - LSTM neural networks
- `optuna >= 3.6.0` - Hyperparameter optimization
- `shap >= 0.42.0` - Model explainability
- `prophet >= 1.1.0` - Time series forecasting

## 📊 Supported SMART Attributes

### Core Attributes
- `smart_5` - Reallocated Sectors Count
- `smart_187` - Reported Uncorrectable Errors
- `smart_197` - Current Pending Sector Count
- `smart_198` - Uncorrectable Sector Count
- `smart_194` - Temperature
- `smart_9` - Power-on Hours
- `smart_12` - Power Cycle Count

### Extended Attributes
- `smart_1` - Read Error Rate
- `smart_3` - Spin-Up Time
- `smart_4` - Start/Stop Count
- `smart_7` - Seek Error Rate
- `smart_10` - Spin Retry Count
- `smart_193` - Load/Unload Cycle Count
- `smart_199` - UltraDMA CRC Error Count
- `smart_240` - Head Flying Hours

## 🎯 Features

### Advanced Feature Engineering
- **Error Metrics**: Error rates, acceleration, burst detection
- **Performance Indicators**: Degradation trends, volatility indices
- **Thermal Analysis**: Temperature variance, stress cycles
- **Mechanical Health**: Stress indices, wear rates, aging factors
- **Cumulative Metrics**: Total wear, reliability degradation

### Risk Scoring
- **Original Ensemble**: 60% XGBoost + 25% Random Forest + 15% RUL
- **Advanced Ensemble**: 7-algorithm soft voting with confidence scoring
- **Multi-dimensional Health**: Error (40%) + Performance (25%) + Thermal (20%) + Mechanical (15%)

### Classification Categories
- **HEALTHY**: Risk score < 0.2
- **MONITOR**: Risk score 0.2-0.5
- **WARN**: Risk score 0.5-0.8
- **FAIL**: Risk score ≥ 0.8

## 📈 Dashboard Features

### Overview Tab
- Fleet health summary
- Risk distribution charts
- Critical drive identification
- Health score projections

### Detailed Analysis Tab
- Feature distribution analysis
- Correlation matrices
- Time series visualizations
- Interactive feature selection

### Anomaly Detection Tab
- Multiple detection methods
- PCA visualization
- Anomaly statistics
- Detailed anomaly reports

### Predictions Tab
- Failure probability distributions
- RUL predictions
- High-risk drive identification
- Confidence interval analysis

### Reports Tab
- Executive summaries
- Health distribution reports
- Critical drive alerts
- Downloadable reports

## 🔧 Configuration

### Model Parameters
```python
# XGBoost parameters
xgb_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Risk thresholds
risk_thresholds = {
    'FAIL': 0.8,
    'WARN': 0.5,
    'MONITOR': 0.2,
    'HEALTHY': 0.0
}
```

### Feature Selection
- Random Forest importance
- F-test statistical significance
- Mutual information scores
- Top-k feature selection

## 📊 Performance Metrics

### Model Evaluation
- **Accuracy**: Overall prediction correctness
- **Precision**: False positive minimization
- **Recall**: False negative minimization
- **F1-Score**: Precision-recall balance
- **AUC-ROC**: Discrimination ability

### Business Metrics
- **False Positive Rate**: Unnecessary replacements
- **False Negative Rate**: Missed failures
- **Mean Time to Detection**: Early warning effectiveness
- **Cost Savings**: Prevented failure costs

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Input    │    │  Feature        │    │   ML Models     │
│                 │    │  Engineering    │    │                 │
│ • CSV Upload    │───▶│ • SMART Attrs   │───▶│ • XGBoost       │
│ • Synthetic     │    │ • Derived Feats │    │ • Random Forest │
│ • Real-time     │    │ • Rolling Stats │    │ • LSTM          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌──────────────────┐           │
│   Dashboard     │    │  Risk Scoring   │◀──────────┘
│                 │    │                 │
│ • Overview      │◀───│ • Ensemble      │
│ • Analysis      │    │ • Health Score  │
│ • Anomalies     │    │ • Classification│
│ • Predictions   │    │                 │
│ • Reports       │    └──────────────────┘
└─────────────────┘
```

## 📚 Documentation

- [**Complete Feature Documentation**](PROJECT_FEATURES.md) - Comprehensive feature and algorithm guide
- [**Architecture Overview**](docs/architecture.md) - System architecture and design
- [**API Reference**](docs/api.md) - Detailed API documentation
- [**User Guide**](docs/user_guide.md) - Step-by-step usage instructions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Backblaze](https://www.backblaze.com/) for SMART data research
- [XGBoost](https://xgboost.ai/) for gradient boosting framework
- [Streamlit](https://streamlit.io/) for the web framework
- [Optuna](https://optuna.org/) for hyperparameter optimization

## 📞 Support

For support and questions:
- Create an [Issue](https://github.com/yourusername/smart-storage-health-monitor/issues)
- Check the [Documentation](PROJECT_FEATURES.md)
- Review the [Examples](examples/)

---

**⚠️ Disclaimer**: This tool is for monitoring and prediction purposes only. Always maintain proper backups and follow data protection best practices.
