"""
Machine Learning models for the Amazon ML Product Pricing Challenge
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from typing import List, Tuple, Dict, Any
import logging
import joblib
import os

logger = logging.getLogger(__name__)

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate SMAPE metric."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Avoid division by zero
    mask = denominator != 0
    smape = np.zeros_like(numerator)
    smape[mask] = numerator[mask] / denominator[mask]
    
    return np.mean(smape) * 100

class PricePredictionModel:
    """Ensemble model for product price prediction."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the price prediction model.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        self.fitted = False
        
        # Initialize individual models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize individual models for the ensemble."""
        
        # Random Forest
        self.models['rf'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # XGBoost
        self.models['xgb'] = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # LightGBM
        self.models['lgb'] = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        # Gradient Boosting
        self.models['gb'] = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        
        # Ridge Regression
        self.models['ridge'] = Ridge(alpha=1.0)
        
        # Elastic Net
        self.models['elastic'] = ElasticNet(
            alpha=1.0,
            l1_ratio=0.5,
            random_state=self.random_state
        )
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Evaluate a single model using cross-validation.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target values
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with evaluation metrics
        """
        kfold = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # SMAPE scores
        smape_scores = []
        mae_scores = []
        rmse_scores = []
        
        for train_idx, val_idx in kfold.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Fit model
            model.fit(X_train_fold, y_train_fold)
            
            # Predict
            y_pred_fold = model.predict(X_val_fold)
            
            # Calculate metrics
            smape_scores.append(calculate_smape(y_val_fold, y_pred_fold))
            mae_scores.append(mean_absolute_error(y_val_fold, y_pred_fold))
            rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_pred_fold)))
        
        return {
            'smape_mean': np.mean(smape_scores),
            'smape_std': np.std(smape_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores)
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PricePredictionModel':
        """
        Fit the ensemble model.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Self
        """
        logger.info("Starting model training...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Evaluate and fit each model
        model_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Evaluate model
            scores = self.evaluate_model(model, X_scaled, y)
            model_scores[name] = scores
            
            logger.info(f"{name} - SMAPE: {scores['smape_mean']:.3f} ± {scores['smape_std']:.3f}")
            
            # Fit on full data
            model.fit(X_scaled, y)
        
        # Calculate ensemble weights based on SMAPE performance
        self._calculate_weights(model_scores)
        
        self.fitted = True
        logger.info("Model training completed")
        
        return self
    
    def _calculate_weights(self, model_scores: Dict[str, Dict[str, float]]):
        """Calculate ensemble weights based on model performance."""
        
        # Use inverse SMAPE as weights (lower SMAPE = higher weight)
        smape_scores = {name: scores['smape_mean'] for name, scores in model_scores.items()}
        
        # Convert to weights (inverse of SMAPE)
        inv_smape = {name: 1.0 / max(score, 0.1) for name, score in smape_scores.items()}
        
        # Normalize weights
        total_weight = sum(inv_smape.values())
        self.weights = {name: weight / total_weight for name, weight in inv_smape.items()}
        
        logger.info("Model weights:")
        for name, weight in self.weights.items():
            logger.info(f"  {name}: {weight:.3f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_scaled)
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'models': self.models,
            'weights': self.weights,
            'scaler': self.scaler,
            'fitted': self.fitted,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.weights = model_data['weights']
        self.scaler = model_data['scaler']
        self.fitted = model_data['fitted']
        self.random_state = model_data['random_state']
        
        logger.info(f"Model loaded from {filepath}")

class AdvancedPricePredictionModel:
    """Advanced ensemble with stacking and feature engineering."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.level1_models = {}
        self.level2_model = None
        self.scaler = StandardScaler()
        self.fitted = False
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models for stacked ensemble."""
        
        # Level 1 models (base models)
        self.level1_models['rf'] = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.level1_models['xgb'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.level1_models['lgb'] = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        
        self.level1_models['gb'] = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state
        )
        
        # Level 2 model (meta-model)
        self.level2_model = Ridge(alpha=0.1)
    
    def _create_level1_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Create level 1 features using cross-validation."""
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        level1_features = np.zeros((X.shape[0], len(self.level1_models)))
        
        for i, (name, model) in enumerate(self.level1_models.items()):
            logger.info(f"Creating level 1 features with {name}...")
            
            for train_idx, val_idx in kfold.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                
                # Train model on fold
                model.fit(X_train_fold, y_train_fold)
                
                # Predict on validation set
                val_pred = model.predict(X_val_fold)
                level1_features[val_idx, i] = val_pred
        
        return level1_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdvancedPricePredictionModel':
        """Fit the stacked ensemble model."""
        
        logger.info("Starting advanced model training...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create level 1 features
        level1_features = self._create_level1_features(X_scaled, y)
        
        # Train level 1 models on full data
        for name, model in self.level1_models.items():
            logger.info(f"Training {name} on full data...")
            model.fit(X_scaled, y)
        
        # Train level 2 model
        logger.info("Training level 2 model...")
        self.level2_model.fit(level1_features, y)
        
        self.fitted = True
        logger.info("Advanced model training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using stacked ensemble."""
        
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get level 1 predictions
        level1_predictions = np.zeros((X.shape[0], len(self.level1_models)))
        
        for i, (name, model) in enumerate(self.level1_models.items()):
            level1_predictions[:, i] = model.predict(X_scaled)
        
        # Get level 2 predictions
        final_predictions = self.level2_model.predict(level1_predictions)
        
        return final_predictions