from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from .advanced_monitoring import AdvancedHealthMonitor


class AdvancedEnsembleModel:
    """Advanced ensemble model with multiple algorithms and feature selection."""
    
    def __init__(self, use_feature_selection: bool = True):
        self.use_feature_selection = use_feature_selection
        self.feature_selector = None
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.monitor = AdvancedHealthMonitor()
        
    def build_ensemble(
        self, 
        features: pd.DataFrame, 
        targets: pd.Series
    ) -> None:
        """Build advanced ensemble with multiple algorithms."""
        # Feature selection
        if self.use_feature_selection:
            features = self._select_important_features(features, targets)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Define base models
        base_models = [
            # Tree-based models
            ('extra_trees', ExtraTreesClassifier(
                n_estimators=200, 
                max_depth=10, 
                random_state=42
            )),
            ('gradient_boost', GradientBoostingClassifier(
                n_estimators=150, 
                learning_rate=0.1, 
                max_depth=6,
                random_state=42
            )),
            ('ada_boost', AdaBoostClassifier(
                n_estimators=100, 
                learning_rate=0.1,
                random_state=42
            )),
            
            # Linear models
            ('logistic', LogisticRegression(
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            )),
            
            # Other models
            ('svm', SVC(
                probability=True, 
                kernel='rbf', 
                random_state=42,
                class_weight='balanced'
            )),
            ('naive_bayes', GaussianNB()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )),
        ]
        
        # Create voting ensemble
        self.ensemble_model = VotingClassifier(
            estimators=base_models,
            voting='soft',  # Use probability averaging
            weights=[2, 2, 1, 1, 1, 1, 1]  # Give more weight to tree models
        )
        
        # Train ensemble
        self.ensemble_model.fit(scaled_features, targets)
    
    def _select_important_features(
        self, 
        features: pd.DataFrame, 
        targets: pd.Series,
        top_k: int = 50
    ) -> pd.DataFrame:
        """Select most important features using multiple methods."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import (
            SelectKBest, 
            f_classif, 
            mutual_info_classif
        )
        
        # Method 1: Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features, targets)
        rf_importance = pd.Series(rf.feature_importances_, index=features.columns)
        
        # Method 2: Statistical tests
        f_scores, _ = f_classif(features, targets)
        f_importance = pd.Series(f_scores, index=features.columns)
        
        # Method 3: Mutual information
        mi_scores = mutual_info_classif(features, targets, random_state=42)
        mi_importance = pd.Series(mi_scores, index=features.columns)
        
        # Combine importance scores
        combined_importance = (
            rf_importance.rank(ascending=False) +
            f_importance.rank(ascending=False) +
            mi_importance.rank(ascending=False)
        )
        
        # Select top features
        top_features = combined_importance.nsmallest(top_k).index
        return features[top_features]
    
    def predict_with_confidence(
        self, 
        features: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Make predictions with confidence scores."""
        if self.use_feature_selection and self.feature_selector is not None:
            features = features[self.feature_selector.get_feature_names_out()]
        
        scaled_features = self.scaler.transform(features)
        
        # Get predictions and probabilities
        predictions = self.ensemble_model.predict(scaled_features)
        probabilities = self.ensemble_model.predict_proba(scaled_features)
        
        # Calculate confidence metrics
        max_proba = np.max(probabilities, axis=1)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        confidence = 1 - (entropy / np.log(len(probabilities[0])))
        
        confidence_metrics = {
            "avg_confidence": np.mean(confidence),
            "avg_max_probability": np.mean(max_proba),
            "prediction_certainty": np.mean(confidence > 0.7),
        }
        
        return predictions, max_proba, confidence_metrics
    
    def predict_ensemble_risk(
        self, 
        features: pd.DataFrame, 
        metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict risk using advanced ensemble and monitoring."""
        # Get ensemble predictions
        predictions, probabilities, confidence_metrics = self.predict_with_confidence(features)
        
        # Get comprehensive health scores
        health_scores = self.monitor.compute_comprehensive_health_score(features, metadata)
        
        # Get failure predictions
        failure_predictions = self.monitor.predict_failure_time(features)
        
        # Combine all predictions
        result = metadata.copy()
        result["ensemble_prediction"] = predictions
        result["ensemble_confidence"] = probabilities
        result["comprehensive_health_score"] = health_scores["comprehensive_health_score"]
        result["health_category"] = health_scores["health_category"]
        result["predicted_rul_days"] = failure_predictions["predicted_rul_days"]
        result["failure_probability"] = failure_predictions["failure_probability"]
        
        # Advanced risk scoring
        result["advanced_risk_score"] = self._calculate_advanced_risk_score(
            probabilities, health_scores["comprehensive_health_score"], 
            failure_predictions["failure_probability"]
        )
        
        # Risk classification
        result["advanced_risk_category"] = pd.cut(
            result["advanced_risk_score"],
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        )
        
        return result
    
    def _calculate_advanced_risk_score(
        self, 
        ensemble_confidence: np.ndarray,
        health_score: pd.Series,
        failure_probability: pd.Series
    ) -> np.ndarray:
        """Calculate advanced risk score combining multiple factors."""
        # Normalize all components to 0-1 range
        normalized_confidence = 1 - ensemble_confidence  # Higher confidence = lower risk
        normalized_health = 1 - health_score  # Higher health = lower risk
        
        # Weighted combination
        risk_score = (
            normalized_confidence * 0.4 +
            normalized_health * 0.4 +
            failure_probability * 0.2
        )
        
        return np.clip(risk_score, 0, 1)
    
    def evaluate_model(
        self, 
        features: pd.DataFrame, 
        targets: pd.Series
    ) -> Dict[str, any]:
        """Comprehensive model evaluation."""
        predictions, probabilities, confidence_metrics = self.predict_with_confidence(features)
        
        # Classification metrics
        report = classification_report(targets, predictions, output_dict=True)
        cm = confusion_matrix(targets, predictions)
        
        # Additional metrics
        accuracy = np.mean(predictions == targets)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "confidence_metrics": confidence_metrics,
        }
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from ensemble models."""
        if not hasattr(self.ensemble_model, 'estimators_'):
            return pd.DataFrame()
        
        importance_data = []
        
        for name, estimator in self.ensemble_model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                importance = estimator.feature_importances_
                importance_data.extend([
                    {"model": name, "feature": feature, "importance": imp}
                    for feature, imp in zip(feature_names, importance)
                ])
        
        return pd.DataFrame(importance_data)
    
    def cross_validate_ensemble(
        self, 
        features: pd.DataFrame, 
        targets: pd.Series,
        cv_folds: int = 5
    ) -> Dict[str, List[float]]:
        """Cross-validate the ensemble model."""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.base import clone
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        scores = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
        }
        
        for train_idx, val_idx in skf.split(features, targets):
            X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = targets.iloc[train_idx], targets.iloc[val_idx]
            
            # Clone and train model
            temp_model = clone(self.ensemble_model)
            temp_scaler = StandardScaler()
            
            X_train_scaled = temp_scaler.fit_transform(X_train)
            X_val_scaled = temp_scaler.transform(X_val)
            
            temp_model.fit(X_train_scaled, y_train)
            y_pred = temp_model.predict(X_val_scaled)
            
            # Calculate scores
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            scores["accuracy"].append(accuracy_score(y_val, y_pred))
            scores["precision"].append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
            scores["recall"].append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
            scores["f1_score"].append(f1_score(y_val, y_pred, average='weighted', zero_division=0))
        
        return {
            metric: {"mean": np.mean(values), "std": np.std(values)}
            for metric, values in scores.items()
        }
