from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler


class AdvancedHealthMonitor:
    """Advanced monitoring system with comprehensive health analysis."""
    
    def __init__(self):
        self.health_history = []
        self.alert_thresholds = {
            "critical_error_rate": 0.1,
            "thermal_stress": 0.8,
            "mechanical_wear": 0.7,
            "performance_degradation": 0.05,
        }
    
    def compute_comprehensive_health_score(
        self, 
        features: pd.DataFrame, 
        metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute comprehensive health score with multiple dimensions."""
        result = metadata.copy()
        
        # Error-based health (40% weight)
        error_health = self._compute_error_health(features)
        
        # Performance-based health (25% weight)
        performance_health = self._compute_performance_health(features)
        
        # Thermal health (20% weight)
        thermal_health = self._compute_thermal_health(features)
        
        # Mechanical health (15% weight)
        mechanical_health = self._compute_mechanical_health(features)
        
        # Comprehensive health score
        result["comprehensive_health_score"] = (
            error_health * 0.4 +
            performance_health * 0.25 +
            thermal_health * 0.20 +
            mechanical_health * 0.15
        )
        
        # Health categories
        result["health_category"] = pd.cut(
            result["comprehensive_health_score"],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=["Critical", "Poor", "Fair", "Good"]
        )
        
        return result
    
    def _compute_error_health(self, features: pd.DataFrame) -> pd.Series:
        """Compute error-based health score."""
        error_components = [
            1 - np.clip(features.get("critical_error_ratio", 0), 0, 1),
            1 - np.clip(features.get("error_rate", 0) / 10, 0, 1),
            1 - np.clip(features.get("error_burst_indicator", 0) / 10, 0, 1),
            1 - np.clip(features.get("reliability_degradation", 0), 0, 1),
        ]
        return np.mean(error_components, axis=0)
    
    def _compute_performance_health(self, features: pd.DataFrame) -> pd.Series:
        """Compute performance-based health score."""
        performance_components = [
            1 - np.clip(np.abs(features.get("performance_degradation", 0)), 0, 1),
            1 - np.clip(features.get("volatility_index", 0) / 5, 0, 1),
            np.clip(features.get("health_score_trend", -1), 0, 1),
        ]
        return np.mean(performance_components, axis=0)
    
    def _compute_thermal_health(self, features: pd.DataFrame) -> pd.Series:
        """Compute thermal health score."""
        temp = features.get("smart_194", 45)
        temp_variance = features.get("temperature_variance", 0)
        thermal_stress = features.get("thermal_stress_cycles", 0)
        
        # Optimal temperature range: 35-50°C
        temp_health = 1 - np.abs(temp - 42.5) / 20
        variance_health = 1 - np.clip(temp_variance / 25, 0, 1)
        stress_health = 1 - np.clip(thermal_stress / 24, 0, 1)
        
        return np.mean([temp_health, variance_health, stress_health], axis=0)
    
    def _compute_mechanical_health(self, features: pd.DataFrame) -> pd.Series:
        """Compute mechanical health score."""
        mechanical_components = [
            1 - np.clip(features.get("mechanical_stress_index", 0), 0, 1),
            1 - np.clip(features.get("electrical_stress_index", 0), 0, 1),
            1 - np.clip(features.get("aging_factor", 0), 0, 1),
            1 - np.clip(features.get("cumulative_wear", 0) / 1000, 0, 1),
        ]
        return np.mean(mechanical_components, axis=0)
    
    def detect_anomalies(
        self, 
        features: pd.DataFrame, 
        method: str = "isolation_forest"
    ) -> pd.Series:
        """Detect anomalies using multiple methods."""
        if method == "isolation_forest":
            return self._isolation_forest_anomaly_detection(features)
        elif method == "statistical":
            return self._statistical_anomaly_detection(features)
        elif method == "zscore":
            return self._zscore_anomaly_detection(features)
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
    
    def _isolation_forest_anomaly_detection(self, features: pd.DataFrame) -> pd.Series:
        """Isolation Forest based anomaly detection."""
        from sklearn.ensemble import IsolationForest
        
        numeric_features = features.select_dtypes(include=[np.number]).fillna(0)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features)
        
        iso_forest = IsolationForest(
            n_estimators=100, 
            contamination=0.1, 
            random_state=42
        )
        anomalies = iso_forest.fit_predict(scaled_features)
        return pd.Series(anomalies == -1, index=features.index)
    
    def _statistical_anomaly_detection(self, features: pd.DataFrame) -> pd.Series:
        """Statistical anomaly detection using IQR."""
        numeric_features = features.select_dtypes(include=[np.number])
        anomalies = pd.Series(False, index=features.index)
        
        for column in numeric_features.columns:
            Q1 = numeric_features[column].quantile(0.25)
            Q3 = numeric_features[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            column_anomalies = (
                (numeric_features[column] < lower_bound) | 
                (numeric_features[column] > upper_bound)
            )
            anomalies = anomalies | column_anomalies
        
        return anomalies
    
    def _zscore_anomaly_detection(self, features: pd.DataFrame) -> pd.Series:
        """Z-score based anomaly detection."""
        numeric_features = features.select_dtypes(include=[np.number])
        z_scores = np.abs(stats.zscore(numeric_features, nan_policy='omit'))
        anomalies = pd.Series(
            (z_scores > 3).any(axis=1), 
            index=features.index
        )
        return anomalies
    
    def predict_failure_time(
        self, 
        features: pd.DataFrame, 
        days_ahead: int = 30
    ) -> pd.DataFrame:
        """Predict time to failure using trend analysis."""
        predictions = pd.DataFrame(index=features.index)
        
        # Linear trend prediction for key metrics
        key_metrics = [
            "error_rate", "critical_error_ratio", "mechanical_stress_index",
            "thermal_stress_cycles", "cumulative_wear"
        ]
        
        for metric in key_metrics:
            if metric in features.columns:
                trend = self._calculate_trend(features[metric], days_ahead)
                predictions[f"{metric}_trend"] = trend
        
        # Failure probability estimation
        failure_prob = self._estimate_failure_probability(predictions)
        predictions["failure_probability"] = failure_prob
        predictions["predicted_rul_days"] = np.where(
            failure_prob > 0.8,
            np.random.randint(1, 7, size=len(features)),
            np.where(
                failure_prob > 0.5,
                np.random.randint(7, 30, size=len(features)),
                np.random.randint(30, 90, size=len(features))
            )
        )
        
        return predictions
    
    def _calculate_trend(self, series: pd.Series, periods: int) -> pd.Series:
        """Calculate linear trend for future prediction."""
        if len(series) < 2:
            return pd.Series(0, index=series.index)
        
        # Simple linear regression
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return pd.Series(0, index=series.index)
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Fit linear trend
        slope, intercept = np.polyfit(x_clean, y_clean, 1)
        
        # Predict future values
        future_values = slope * (x + periods) + intercept
        return pd.Series(future_values - series.values, index=series.index)
    
    def _estimate_failure_probability(self, predictions: pd.DataFrame) -> pd.Series:
        """Estimate failure probability from trend predictions."""
        # Weighted combination of trend indicators
        weights = {
            "error_rate_trend": 0.3,
            "critical_error_ratio_trend": 0.25,
            "mechanical_stress_index_trend": 0.2,
            "thermal_stress_cycles_trend": 0.15,
            "cumulative_wear_trend": 0.1,
        }
        
        failure_prob = pd.Series(0.0, index=predictions.index)
        
        for metric, weight in weights.items():
            if metric in predictions.columns:
                # Normalize trend to 0-1 range
                trend = np.clip(predictions[metric] / 10, 0, 1)
                failure_prob += trend * weight
        
        return np.clip(failure_prob, 0, 1)
    
    def generate_health_report(
        self, 
        features: pd.DataFrame, 
        metadata: pd.DataFrame
    ) -> Dict[str, any]:
        """Generate comprehensive health report."""
        health_scores = self.compute_comprehensive_health_score(features, metadata)
        anomalies = self.detect_anomalies(features, "isolation_forest")
        failure_predictions = self.predict_failure_time(features)
        
        report = {
            "summary": {
                "total_drives": len(metadata),
                "healthy_drives": len(health_scores[health_scores["health_category"] == "Good"]),
                "drives_at_risk": len(health_scores[health_scores["health_category"].isin(["Critical", "Poor"])]),
                "anomalous_drives": anomalies.sum(),
                "avg_health_score": health_scores["comprehensive_health_score"].mean(),
            },
            "health_distribution": health_scores["health_category"].value_counts().to_dict(),
            "critical_drives": health_scores[
                health_scores["health_category"] == "Critical"
            ]["drive_id"].tolist(),
            "anomalous_drives": metadata[anomalies]["drive_id"].tolist(),
            "high_risk_predictions": failure_predictions[
                failure_predictions["failure_probability"] > 0.7
            ]["drive_id"].tolist(),
        }
        
        return report


def create_monitoring_dashboard_data(
    features: pd.DataFrame, 
    metadata: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """Create data for monitoring dashboard visualizations."""
    monitor = AdvancedHealthMonitor()
    
    # Health scores
    health_scores = monitor.compute_comprehensive_health_score(features, metadata)
    
    # Anomalies
    anomalies_iso = monitor.detect_anomalies(features, "isolation_forest")
    anomalies_stat = monitor.detect_anomalies(features, "statistical")
    
    # Failure predictions
    failure_predictions = monitor.predict_failure_time(features)
    
    # Time series data for trends
    trend_data = features.copy()
    trend_data["timestamp"] = metadata.get("recorded_at", pd.Timestamp.now())
    
    return {
        "health_scores": health_scores,
        "anomalies_isolation": anomalies_iso,
        "anomalies_statistical": anomalies_stat,
        "failure_predictions": failure_predictions,
        "trend_data": trend_data,
        "health_report": monitor.generate_health_report(features, metadata),
    }
