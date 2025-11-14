"""
Memory-Efficient Optimized Amazon ML Model
"""
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
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
    """Enhanced quantity extraction - most predictive feature"""
    if pd.isna(text):
        return 1.0
    
    text = str(text).lower()
    
    # Value field pattern (highest priority)
    value_match = re.search(r'value:\s*(\d+\.?\d*)', text)
    if value_match:
        return float(value_match.group(1))
    
    # Pack quantity patterns
    patterns = [
        r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)',  # Multi-pack
        r'pack\s*of\s*(\d+)',
        r'(\d+)\s*pack',
        r'case\s*of\s*(\d+)',
        r'(\d+\.?\d*)\s*(count|ct|piece)',
    ]
    
    quantities = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                nums = [float(x) for x in match if x.replace('.', '').isdigit()]
                if len(nums) == 2:
                    quantities.append(nums[0] * nums[1])
                else:
                    quantities.extend(nums)
            else:
                quantities.append(float(match))
    
    return max(quantities) if quantities else 1.0

def create_efficient_features(df):
    """Create memory-efficient feature set"""
    
    # Minimal text preprocessing
    processed_text = df['catalog_content'].fillna('').apply(
        lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', str(x).lower())
    )
    
    # Reduced TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=500,  # Reduced from 1000
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        stop_words='english'
    )
    tfidf_features = tfidf.fit_transform(processed_text).toarray()
    
    # Essential manual features only
    manual_features = []
    for _, row in df.iterrows():
        text = str(row['catalog_content']).lower()
        
        quantity = extract_quantity_advanced(row['catalog_content'])
        
        features = {
            'quantity': quantity,
            'text_length': len(str(row['catalog_content'])),
            'word_count': len(str(row['catalog_content']).split()),
            'premium_score': sum(1 for w in ['premium', 'gourmet', 'organic', 'natural'] if w in text),
            'pack_score': sum(1 for w in ['pack', 'case', 'box', 'jar'] if w in text),
            'size_score': sum(1 for w in ['large', 'jumbo', 'family', 'bulk'] if w in text),
            'health_score': sum(1 for w in ['healthy', 'gluten free', 'vegan'] if w in text),
        }
        
        # Price per unit if available
        if 'price' in row:
            features['price_per_unit'] = row['price'] / max(quantity, 1)
        
        manual_features.append(features)
    
    manual_df = pd.DataFrame(manual_features)
    
    # Combine features
    combined_features = np.hstack([tfidf_features, manual_df.values])
    
    # Feature selection to reduce dimensionality
    if 'price' in df.columns:
        selector = SelectKBest(f_regression, k=min(300, combined_features.shape[1]))
        combined_features = selector.fit_transform(combined_features, df['price'])
        print(f"Selected {combined_features.shape[1]} best features")
    
    return combined_features, tfidf, manual_df.columns.tolist()

class EfficientEnsemble:
    """Memory-efficient ensemble"""
    
    def __init__(self):
        # Smaller, more efficient models
        self.models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=200,  # Reduced
                max_depth=6,       # Reduced
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                tree_method='hist'  # More memory efficient
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=100,  # Reduced
                max_depth=10,      # Reduced
                min_samples_split=5,
                random_state=42,
                n_jobs=2  # Limit parallel jobs
            )
        }
        
        self.meta_model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X, y):
        print(f"Training efficient ensemble on {X.shape[1]} features...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # 3-fold CV for speed
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        oof_preds = np.zeros((X.shape[0], len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            print(f"Training {name}...")
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                oof_preds[val_idx, i] = val_pred
                
                fold_score = calculate_smape(y_val, val_pred)
                fold_scores.append(fold_score)
            
            avg_score = np.mean(fold_scores)
            print(f"{name} CV SMAPE: {avg_score:.3f}")
        
        # Train meta-model
        self.meta_model.fit(oof_preds, y)
        
        # Retrain on full data
        for model in self.models.values():
            model.fit(X_scaled, y)
        
        # Final ensemble score
        final_pred = self.meta_model.predict(oof_preds)
        ensemble_score = calculate_smape(y, final_pred)
        print(f"Ensemble CV SMAPE: {ensemble_score:.3f}")
        
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
    print("Starting memory-efficient Amazon ML model...")
    
    # Load data
    try:
        train_df = pd.read_csv('dataset/train.csv')
        test_df = pd.read_csv('dataset/test.csv')
        print(f"Loaded {len(train_df)} train, {len(test_df)} test samples")
    except:
        print("Using sample data...")
        sample_test = pd.read_csv('dataset/sample_test.csv')
        sample_out = pd.read_csv('dataset/sample_test_out.csv')
        train_df = sample_test.merge(sample_out, on='sample_id')
        test_df = sample_test.copy()
        
        # Augment sample data for better training
        augmented_data = []
        for _ in range(5):
            aug_df = train_df.copy()
            noise = np.random.normal(1, 0.1, len(aug_df))
            aug_df['price'] = aug_df['price'] * noise
            augmented_data.append(aug_df)
        
        train_df = pd.concat([train_df] + augmented_data, ignore_index=True)
        print(f"Augmented to {len(train_df)} training samples")
    
    # Outlier treatment
    if 'price' in train_df.columns:
        Q1, Q3 = train_df['price'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        train_df['price'] = train_df['price'].clip(lower, upper)
    
    # Create features
    print("Creating efficient features...")
    X_train, tfidf, manual_cols = create_efficient_features(train_df)
    y_train = train_df['price'].values
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # Train model
    model = EfficientEnsemble()
    model.fit(X_train, y_train)
    
    # Predict on test
    print("Making predictions...")
    X_test, _, _ = create_efficient_features(test_df)
    predictions = model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    submission.to_csv('efficient_submission.csv', index=False)
    print(f"Efficient submission saved!")
    print(f"Prediction stats: Min={predictions.min():.2f}, Max={predictions.max():.2f}, Mean={predictions.mean():.2f}")

if __name__ == "__main__":
    main()