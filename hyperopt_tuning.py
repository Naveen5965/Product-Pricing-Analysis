"""
Hyperparameter optimization for Amazon ML model
"""
import pandas as pd
import numpy as np
from improved_model import create_advanced_features, calculate_smape
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
import xgboost as xgb
import lightgbm as lgb
import optuna
import warnings
warnings.filterwarnings('ignore')

def objective_xgb(trial, X, y):
    """Objective function for XGBoost optimization"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 800),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**params)
    
    # 3-fold CV for speed
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = calculate_smape(y_val, pred)
        scores.append(score)
    
    return np.mean(scores)

def objective_lgb(trial, X, y):
    """Objective function for LightGBM optimization"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 800),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'random_state': 42,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = calculate_smape(y_val, pred)
        scores.append(score)
    
    return np.mean(scores)

def optimize_hyperparameters():
    """Run hyperparameter optimization"""
    print("Loading data for hyperparameter optimization...")
    
    # Load data
    try:
        train_df = pd.read_csv('dataset/train.csv')
        print(f"Loaded {len(train_df)} training samples")
    except:
        print("Using sample data...")
        sample_test = pd.read_csv('dataset/sample_test.csv')
        sample_out = pd.read_csv('dataset/sample_test_out.csv')
        train_df = sample_test.merge(sample_out, on='sample_id')
    
    # Create features
    X_train, _ = create_advanced_features(train_df)
    y_train = train_df['price'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    print("Starting XGBoost optimization...")
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_scaled, y_train), n_trials=50)
    
    print("Best XGBoost parameters:")
    print(study_xgb.best_params)
    print(f"Best XGBoost SMAPE: {study_xgb.best_value:.3f}")
    
    print("\nStarting LightGBM optimization...")
    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(lambda trial: objective_lgb(trial, X_scaled, y_train), n_trials=50)
    
    print("Best LightGBM parameters:")
    print(study_lgb.best_params)
    print(f"Best LightGBM SMAPE: {study_lgb.best_value:.3f}")
    
    # Save best parameters
    best_params = {
        'xgb': study_xgb.best_params,
        'lgb': study_lgb.best_params
    }
    
    import json
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print("Best parameters saved to 'best_params.json'")

if __name__ == "__main__":
    optimize_hyperparameters()