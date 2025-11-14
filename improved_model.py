"""
Improved Amazon ML Product Pricing Model - Phase 1 Implementation
Focus: Enhanced feature engineering, advanced ensemble, hyperparameter optimization
"""
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    smape = np.zeros_like(numerator)
    smape[mask] = numerator[mask] / denominator[mask]
    return np.mean(smape) * 100

def advanced_quantity_extraction(text):
    """Enhanced quantity extraction with better patterns"""
    if pd.isna(text):
        return 1.0
    
    text = str(text).lower()
    patterns = [
        r'value:\s*(\d+\.?\d*)',  # Most important - Value field
        r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)',  # Multi-pack patterns
        r'(\d+\.?\d*)\s*(pack|count|ct|piece|pc|pcs)',
        r'(\d+\.?\d*)\s*(oz|ounce|fl\s*oz|fluid\s*ounce)',
        r'(\d+\.?\d*)\s*(pound|lb|gram|g|kg)',
        r'(\d+\.?\d*)\s*(liter|l|ml|gallon|qt|quart)',
        r'pack\s*of\s*(\d+)',
        r'(\d+)\s*-\s*(pack|count)',
        r'case\s*of\s*(\d+)',
    ]
    
    quantities = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                nums = [float(x) for x in match if x.replace('.', '').isdigit()]
                if len(nums) == 2:  # For patterns like "12 x 16"
                    quantities.append(nums[0] * nums[1])
                else:
                    quantities.extend(nums)
            else:
                quantities.append(float(match))
    
    return max(quantities) if quantities else 1.0

def extract_enhanced_features(df):
    """Extract comprehensive features from catalog content"""
    features = []
    
    for _, row in df.iterrows():
        text = str(row['catalog_content']).lower()
        feature_dict = {}
        
        # Basic text stats
        feature_dict['text_length'] = len(str(row['catalog_content']))
        feature_dict['word_count'] = len(str(row['catalog_content']).split())
        
        # Enhanced quantity extraction
        feature_dict['quantity'] = advanced_quantity_extraction(row['catalog_content'])
        
        # Price per unit features
        feature_dict['price_per_unit'] = row.get('price', 0) / max(feature_dict['quantity'], 1)
        
        # Brand indicators (common food brands)
        brands = ['kraft', 'nestle', 'pepsi', 'coca', 'kellogg', 'general mills', 'unilever', 'procter']
        feature_dict['has_brand'] = sum(1 for brand in brands if brand in text)
        
        # Premium indicators
        premium_words = ['premium', 'gourmet', 'luxury', 'artisan', 'organic', 'natural']
        feature_dict['premium_score'] = sum(1 for word in premium_words if word in text)
        
        # Health indicators
        health_words = ['healthy', 'gluten free', 'vegan', 'organic', 'natural', 'vitamin']
        feature_dict['health_score'] = sum(1 for word in health_words if word in text)
        
        # Package type indicators
        package_words = ['pack', 'case', 'box', 'jar', 'bottle', 'bag', 'can']
        feature_dict['package_score'] = sum(1 for word in package_words if word in text)
        
        # Size indicators
        size_words = ['large', 'big', 'jumbo', 'family', 'bulk', 'mega', 'super']
        feature_dict['size_score'] = sum(1 for word in size_words if word in text)
        
        # Category detection
        categories = {
            'snack': ['chip', 'cracker', 'cookie', 'candy', 'chocolate'],
            'beverage': ['drink', 'juice', 'water', 'soda', 'coffee', 'tea'],
            'food': ['sauce', 'soup', 'pasta', 'rice', 'bean', 'meat'],
            'spice': ['spice', 'seasoning', 'salt', 'pepper', 'herb']
        }
        
        for cat, words in categories.items():
            feature_dict[f'is_{cat}'] = sum(1 for word in words if word in text)
        
        features.append(feature_dict)
    
    return pd.DataFrame(features)

class AdvancedEnsemble:
    """Advanced ensemble with cross-validation and multiple models"""
    
    def __init__(self):
        self.base_models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'lgb': lgb.LGBMRegressor(
                n_estimators=500, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=300, max_depth=15, min_samples_split=5,
                random_state=42, n_jobs=-1
            ),
            'et': ExtraTreesRegressor(
                n_estimators=300, max_depth=15, min_samples_split=5,
                random_state=42, n_jobs=-1
            )
        }
        
        self.meta_model = Ridge(alpha=10.0)
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X, y, cv_folds=5):
        """Fit ensemble with cross-validation"""
        print("Training advanced ensemble...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create out-of-fold predictions
        oof_predictions = np.zeros((X.shape[0], len(self.base_models)))
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"Training {name}...")
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train_fold, y_val = y[train_idx], y[val_idx]
                
                # Fit model
                model.fit(X_train, y_train_fold)
                
                # Predict on validation
                val_pred = model.predict(X_val)
                oof_predictions[val_idx, i] = val_pred
                
                # Calculate fold score
                fold_score = calculate_smape(y_val, val_pred)
                fold_scores.append(fold_score)
            
            avg_score = np.mean(fold_scores)
            print(f"{name} CV SMAPE: {avg_score:.3f} ± {np.std(fold_scores):.3f}")
        
        # Train meta-model
        print("Training meta-model...")
        self.meta_model.fit(oof_predictions, y)
        
        # Retrain base models on full data
        for model in self.base_models.values():
            model.fit(X_scaled, y)
        
        self.fitted = True
        
        # Calculate ensemble CV score
        ensemble_pred = self.meta_model.predict(oof_predictions)
        ensemble_score = calculate_smape(y, ensemble_pred)
        print(f"Ensemble CV SMAPE: {ensemble_score:.3f}")
        
        return self
    
    def predict(self, X):
        """Make predictions using ensemble"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models.values()):
            base_predictions[:, i] = model.predict(X_scaled)
        
        # Meta-model prediction
        final_pred = self.meta_model.predict(base_predictions)
        return np.maximum(final_pred, 0.1)  # Ensure positive predictions

def create_advanced_features(df):
    """Create comprehensive feature set"""
    print("Creating advanced features...")
    
    # Text preprocessing
    def preprocess_text(text):
        if pd.isna(text):
            return ""
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        return ' '.join(text.lower().split())
    
    processed_text = df['catalog_content'].apply(preprocess_text)
    
    # Multiple TF-IDF configurations
    tfidf_configs = [
        {'max_features': 1500, 'ngram_range': (1, 2), 'min_df': 2, 'max_df': 0.95},
        {'max_features': 500, 'ngram_range': (1, 3), 'min_df': 3, 'max_df': 0.9},
    ]
    
    all_tfidf_features = []
    for i, config in enumerate(tfidf_configs):
        tfidf = TfidfVectorizer(**config, stop_words='english')
        tfidf_features = tfidf.fit_transform(processed_text).toarray()
        all_tfidf_features.append(tfidf_features)
    
    # Combine TF-IDF features
    combined_tfidf = np.hstack(all_tfidf_features)
    
    # Manual features
    manual_features = extract_enhanced_features(df)
    
    # Combine all features
    final_features = np.hstack([combined_tfidf, manual_features.values])
    
    print(f"Created {final_features.shape[1]} features")
    return final_features, manual_features.columns.tolist()

def main():
    """Main training pipeline"""
    print("Starting improved Amazon ML model training...")
    
    # Load data
    try:
        train_df = pd.read_csv('dataset/train.csv')
        test_df = pd.read_csv('dataset/test.csv')
        print(f"Loaded {len(train_df)} training samples, {len(test_df)} test samples")
    except:
        print("Using sample data...")
        sample_test = pd.read_csv('dataset/sample_test.csv')
        sample_out = pd.read_csv('dataset/sample_test_out.csv')
        train_df = sample_test.merge(sample_out, on='sample_id')
        test_df = sample_test.copy()
    
    # Create features
    X_train, feature_names = create_advanced_features(train_df)
    y_train = train_df['price'].values
    
    # Train advanced ensemble
    model = AdvancedEnsemble()
    model.fit(X_train, y_train)
    
    # Make predictions on test data
    print("Making test predictions...")
    X_test, _ = create_advanced_features(test_df)
    test_predictions = model.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': test_predictions
    })
    
    submission.to_csv('improved_submission.csv', index=False)
    print("Improved submission saved to 'improved_submission.csv'")
    
    # Print prediction statistics
    print(f"Prediction stats - Min: {test_predictions.min():.2f}, "
          f"Max: {test_predictions.max():.2f}, Mean: {test_predictions.mean():.2f}")

if __name__ == "__main__":
    main()