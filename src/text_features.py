"""
Text feature extraction for the Amazon ML Product Pricing Challenge
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
import re
import nltk
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class TextFeatureExtractor:
    """Extract features from product catalog content."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple = (1, 2)):
        """
        Initialize text feature extractor.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: N-gram range for TF-IDF
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # Initialize vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=1000,
            ngram_range=(1, 1),
            stop_words='english'
        )
        
        # LDA for topic modeling
        self.lda_model = LatentDirichletAllocation(
            n_components=20,
            random_state=42,
            max_iter=10
        )
        
        self.scaler = StandardScaler()
        self.fitted = False
        
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove HTML-like tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove extra punctuation but keep important ones
        text = re.sub(r'[^\w\s\-\.\,\!\?]', ' ', text)
        
        return text.strip()
    
    def extract_manual_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Extract manual features from text.
        
        Args:
            texts: List of texts
            
        Returns:
            DataFrame with manual features
        """
        features = []
        
        for text in texts:
            if pd.isna(text):
                text = ""
            
            text_lower = str(text).lower()
            feature_dict = {}
            
            # Basic text statistics
            feature_dict['text_length'] = len(str(text))
            feature_dict['word_count'] = len(str(text).split())
            feature_dict['unique_word_count'] = len(set(str(text).lower().split()))
            feature_dict['avg_word_length'] = np.mean([len(word) for word in str(text).split()]) if str(text).split() else 0
            
            # Punctuation counts
            feature_dict['exclamation_count'] = text.count('!')
            feature_dict['question_count'] = text.count('?')
            feature_dict['comma_count'] = text.count(',')
            feature_dict['period_count'] = text.count('.')
            
            # Product quality indicators
            quality_words = ['premium', 'gourmet', 'luxury', 'deluxe', 'superior', 'finest', 'artisan']
            feature_dict['quality_words'] = sum(1 for word in quality_words if word in text_lower)
            
            # Organic/Natural indicators
            organic_words = ['organic', 'natural', 'pure', 'raw', 'unprocessed']
            feature_dict['organic_words'] = sum(1 for word in organic_words if word in text_lower)
            
            # Health indicators
            health_words = ['healthy', 'nutritious', 'vitamin', 'protein', 'fiber', 'antioxidant']
            feature_dict['health_words'] = sum(1 for word in health_words if word in text_lower)
            
            # Diet indicators
            diet_words = ['gluten free', 'vegan', 'vegetarian', 'kosher', 'halal', 'keto', 'paleo']
            feature_dict['diet_words'] = sum(1 for word in diet_words if word in text_lower)
            
            # Packaging indicators
            pack_words = ['pack', 'case', 'box', 'jar', 'bottle', 'bag', 'container']
            feature_dict['pack_words'] = sum(1 for word in pack_words if word in text_lower)
            
            # Brand/certification indicators
            cert_words = ['certified', 'approved', 'verified', 'tested', 'guaranteed']
            feature_dict['cert_words'] = sum(1 for word in cert_words if word in text_lower)
            
            # Extract quantities and units
            feature_dict.update(self._extract_quantity_features(text))
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _extract_quantity_features(self, text: str) -> Dict:
        """Extract quantity-related features from text."""
        features = {}
        text_lower = str(text).lower()
        
        # Weight patterns
        weight_pattern = r'(\d+\.?\d*)\s*(oz|ounce|lb|pound|g|gram|kg|kilogram)'
        weight_matches = re.findall(weight_pattern, text_lower)
        
        # Volume patterns
        volume_pattern = r'(\d+\.?\d*)\s*(fl\s*oz|fluid\s*ounce|ml|liter|l|gallon|qt|quart)'
        volume_matches = re.findall(volume_pattern, text_lower)
        
        # Count patterns
        count_pattern = r'(\d+\.?\d*)\s*(count|pack|piece|item)'
        count_matches = re.findall(count_pattern, text_lower)
        
        # Extract maximum values
        features['max_weight'] = max([float(match[0]) for match in weight_matches], default=0)
        features['max_volume'] = max([float(match[0]) for match in volume_matches], default=0)
        features['max_count'] = max([float(match[0]) for match in count_matches], default=1)
        
        # Count different types of measurements
        features['weight_mentions'] = len(weight_matches)
        features['volume_mentions'] = len(volume_matches)
        features['count_mentions'] = len(count_matches)
        
        # Extract value field
        value_pattern = r'value:\s*(\d+\.?\d*)'
        value_match = re.search(value_pattern, text_lower)
        features['value_field'] = float(value_match.group(1)) if value_match else 0
        
        return features
    
    def fit(self, texts: List[str]) -> 'TextFeatureExtractor':
        """
        Fit the feature extractors.
        
        Args:
            texts: List of texts to fit on
            
        Returns:
            Self
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Fit TF-IDF
        logger.info("Fitting TF-IDF vectorizer...")
        self.tfidf_vectorizer.fit(processed_texts)
        
        # Fit count vectorizer for LDA
        logger.info("Fitting Count vectorizer for LDA...")
        count_features = self.count_vectorizer.fit_transform(processed_texts)
        
        # Fit LDA
        logger.info("Fitting LDA model...")
        self.lda_model.fit(count_features)
        
        # Fit scaler for manual features
        manual_features = self.extract_manual_features(texts)
        self.scaler.fit(manual_features)
        
        self.fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to features.
        
        Args:
            texts: List of texts to transform
            
        Returns:
            Feature matrix
        """
        if not self.fitted:
            raise ValueError("Must fit before transform")
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # TF-IDF features
        logger.info("Extracting TF-IDF features...")
        tfidf_features = self.tfidf_vectorizer.transform(processed_texts).toarray()
        
        # LDA features
        logger.info("Extracting LDA features...")
        count_features = self.count_vectorizer.transform(processed_texts)
        lda_features = self.lda_model.transform(count_features)
        
        # Manual features
        logger.info("Extracting manual features...")
        manual_features = self.extract_manual_features(texts)
        manual_features_scaled = self.scaler.transform(manual_features)
        
        # Combine all features
        all_features = np.hstack([
            tfidf_features,
            lda_features,
            manual_features_scaled
        ])
        
        logger.info(f"Extracted {all_features.shape[1]} text features")
        return all_features
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Feature matrix
        """
        return self.fit(texts).transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if not self.fitted:
            raise ValueError("Must fit before getting feature names")
        
        feature_names = []
        
        # TF-IDF feature names
        feature_names.extend([f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()])
        
        # LDA feature names
        feature_names.extend([f"lda_topic_{i}" for i in range(self.lda_model.n_components)])
        
        # Manual feature names
        manual_features = self.extract_manual_features(["sample text"])
        feature_names.extend([f"manual_{name}" for name in manual_features.columns])
        
        return feature_names