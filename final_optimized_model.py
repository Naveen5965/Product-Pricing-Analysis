"""
Final Optimized Amazon ML Model - All Phase 1 Improvements
"""
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def calculate_smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    smape = np.zeros_like(numerator)
    smape[mask] = numerator[mask] / denominator[mask]
    return np.mean(smape) * 100

def extract_quantity_advanced(text):
    """Most important feature - enhanced quantity extraction"""
    if pd.isna(text):
        return 1.0
    
    text = str(text).lower()
    
    # Priority patterns - Value field is most predictive
    value_match = re.search(r'value:\s*(\d+\.?\d*)', text)
    if value_match:
        return float(value_match.group(1))
    
    # Multi-pack patterns
    multi_patterns = [
        r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)',
        r'(\d+)\s*pack.*?(\d+\.?\d*)\s*(oz|ounce)',
    ]
    
    for pattern in multi_patterns:
        match = re.search(pattern, text)
        if match:
            nums = [float(x) for x in match.groups() if x.replace('.', '').isdigit()]
            if len(nums) >= 2:
                return nums[0] * nums[1]
    
    # Single quantity patterns
    single_patterns = [
        r'(\d+\.?\d*)\s*(pack|count|ct|piece)',
        r'pack\s*of\s*(\d+)',
        r'case\s*of\s*(\d+)',
        r'(\d+)\s*-\s*pack',
    ]
    
    quantities = []
    for pattern in single_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                quantities.extend([float(x) for x in match if x.replace('.', '').isdigit()])
            else:
                quantities.append(float(match))
    
    return max(quantities) if quantities else 1.0

def create_optimized_features(df):
    """Create optimized feature set focusing on most predictive features"""
    
    # Text preprocessing
    processed_text = df['catalog_content'].fillna('').apply(
        lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', str(x).lower())
    )
    
    # Optimized TF-IDF - reduced features to save memory
    tfidf = TfidfVectorizer(
        max_features=500,     # Reduced from 1000
        ngram_range=(1, 2),
        min_df=3,             # Increased from 2
        max_df=0.9,           # Reduced from 0.95
        stop_words='english'
    )
    tfidf_features = tfidf.fit_transform(processed_text).toarray()
    
    # Manual features - focus on most predictive ones
    manual_features = []
    for _, row in df.iterrows():
        text = str(row['catalog_content']).lower()
        
        features = {
            'quantity': extract_quantity_advanced(row['catalog_content']),
            'text_length': len(str(row['catalog_content'])),
            'word_count': len(str(row['catalog_content']).split()),
        }
        
        # Price per unit (if price available)
        if 'price' in row:
            features['price_per_unit'] = row['price'] / max(features['quantity'], 1)
        
        # High-impact categorical features
        features['premium_words'] = sum(1 for w in ['premium', 'gourmet', 'organic'] if w in text)
        features['pack_words'] = sum(1 for w in ['pack', 'case', 'box'] if w in text)
        features['size_words'] = sum(1 for w in ['large', 'jumbo', 'family'] if w in text)
        
        manual_features.append(features)
    
    manual_df = pd.DataFrame(manual_features)
    
    # Combine features
    combined_features = np.hstack([tfidf_features, manual_df.values])
    
    return combined_features, tfidf, manual_df.columns.tolist()

class OptimizedEnsemble:
    """Memory-optimized ensemble with reduced complexity"""
    
    def __init__(self):
        # Memory-optimized parameters to prevent allocation errors
        self.models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=50,     # Reduced from 500
                max_depth=4,         # Reduced from 8
                learning_rate=0.1,   # Increased to compensate
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                reg_lambda=1,
                random_state=42,
                n_jobs=1,            # Single thread to save memory
                tree_method='hist'   # More memory efficient
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=50,     # Reduced from 500
                max_depth=4,         # Reduced from 8
                learning_rate=0.1,   # Increased to compensate
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=1,
                reg_lambda=1,
                random_state=42,
                verbose=-1,
                n_jobs=1             # Single thread to save memory
            ),
            'rf': RandomForestRegressor(
                n_estimators=50,     # Reduced from 200
                max_depth=6,         # Reduced from 12
                min_samples_split=10,
                random_state=42,
                n_jobs=1             # Single thread to save memory
            )
        }
        
        self.meta_model = Ridge(alpha=5.0)
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X, y):
        print("Training memory-optimized ensemble...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Reduced CV for memory efficiency
        kf = KFold(n_splits=3, shuffle=True, random_state=42)  # Reduced from 5
        oof_preds = np.zeros((X.shape[0], len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            print(f"Training {name}...")
            
            for train_idx, val_idx in kf.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train = y[train_idx]
                
                model.fit(X_train, y_train)
                oof_preds[val_idx, i] = model.predict(X_val)
        
        # Train meta-model
        self.meta_model.fit(oof_preds, y)
        
        # Retrain on full data
        for model in self.models.values():
            model.fit(X_scaled, y)
        
        # Calculate CV score
        final_pred = self.meta_model.predict(oof_preds)
        cv_score = calculate_smape(y, final_pred)
        print(f"Memory-optimized ensemble CV SMAPE: {cv_score:.3f}")
        
        self.fitted = True
        return self
    
    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        
        # Get base predictions
        base_preds = np.column_stack([
            model.predict(X_scaled) for model in self.models.values()
        ])
        
        # Meta prediction
        final_pred = self.meta_model.predict(base_preds)
        return np.maximum(final_pred, 0.01)

def main():
    print("Starting memory-optimized Amazon ML model...")
    
    # Load data
    try:
        train_df = pd.read_csv('dataset/train.csv')
        test_df = pd.read_csv('dataset/test.csv')
        print(f"Loaded {len(train_df)} train, {len(test_df)} test samples")
        
        # Memory optimization - use subset if dataset is too large
        if len(train_df) > 15000:
            print("Large dataset detected, using subset for memory efficiency...")
            train_df = train_df.sample(n=15000, random_state=42).reset_index(drop=True)
            
    except:
        print("Using sample data...")
        sample_test = pd.read_csv('dataset/sample_test.csv')
        sample_out = pd.read_csv('dataset/sample_test_out.csv')
        train_df = sample_test.merge(sample_out, on='sample_id')
        test_df = sample_test.copy()
    
    # Outlier treatment - cap extreme prices
    if 'price' in train_df.columns:
        Q1, Q3 = train_df['price'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        train_df['price'] = train_df['price'].clip(lower, upper)
    
    # Create features for BOTH train and test together to ensure consistency
    print("Creating optimized features...")
    
    # Combine train and test for consistent TF-IDF vocabulary
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    X_combined, tfidf, manual_cols = create_optimized_features(combined_df)
    
    # Split back into train and test
    n_train = len(train_df)
    X_train = X_combined[:n_train]
    X_test = X_combined[n_train:]
    y_train = train_df['price'].values
    
    print(f"Train feature matrix shape: {X_train.shape}")
    print(f"Test feature matrix shape: {X_test.shape}")
    print(f"Memory usage: ~{X_train.nbytes / (1024**2):.1f} MB")
    
    # Train model
    model = OptimizedEnsemble()
    model.fit(X_train, y_train)
    
    # Predict on test
    print("Making predictions...")
    predictions = model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    submission.to_csv('memory_optimized_submission.csv', index=False)
    print(f"Memory-optimized submission saved! Prediction range: {predictions.min():.2f} - {predictions.max():.2f}")
    print("Model completed successfully with reduced memory usage.")

if __name__ == "__main__":
    main()