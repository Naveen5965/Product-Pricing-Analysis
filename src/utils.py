"""
Utility functions for the Amazon ML Product Pricing Challenge
"""
import requests
import os
import time
from typing import List, Optional
import pandas as pd
from PIL import Image
import numpy as np
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_images(df: pd.DataFrame, 
                   image_dir: str = "images", 
                   batch_size: int = 50,
                   max_retries: int = 3,
                   delay: float = 0.1) -> List[str]:
    """
    Download images from URLs in the dataframe.
    
    Args:
        df: DataFrame containing 'sample_id' and 'image_link' columns
        image_dir: Directory to save images
        batch_size: Number of images to download in each batch
        max_retries: Maximum number of retry attempts for failed downloads
        delay: Delay between downloads to avoid throttling
        
    Returns:
        List of successfully downloaded image paths
    """
    os.makedirs(image_dir, exist_ok=True)
    downloaded_paths = []
    failed_downloads = []
    
    total_images = len(df)
    logger.info(f"Starting download of {total_images} images...")
    
    for idx, row in df.iterrows():
        sample_id = row['sample_id']
        image_url = row['image_link']
        
        if pd.isna(image_url) or not image_url:
            logger.warning(f"No image URL for sample {sample_id}")
            continue
            
        # Create filename
        image_path = os.path.join(image_dir, f"{sample_id}.jpg")
        
        # Skip if already downloaded
        if os.path.exists(image_path):
            downloaded_paths.append(image_path)
            continue
            
        # Attempt download with retries
        success = False
        for retry in range(max_retries):
            try:
                response = requests.get(image_url, timeout=30, 
                                      headers={'User-Agent': 'Mozilla/5.0'})
                response.raise_for_status()
                
                # Verify it's an image
                img = Image.open(BytesIO(response.content))
                img.verify()
                
                # Save image
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                    
                downloaded_paths.append(image_path)
                success = True
                logger.info(f"Downloaded {idx+1}/{total_images}: {sample_id}")
                break
                
            except Exception as e:
                logger.warning(f"Retry {retry+1} failed for {sample_id}: {str(e)}")
                time.sleep(delay * (retry + 1))  # Exponential backoff
                
        if not success:
            failed_downloads.append(sample_id)
            logger.error(f"Failed to download image for sample {sample_id}")
            
        # Throttling
        time.sleep(delay)
        
        # Progress update
        if (idx + 1) % batch_size == 0:
            logger.info(f"Progress: {idx+1}/{total_images} processed")
    
    logger.info(f"Download complete: {len(downloaded_paths)} successful, {len(failed_downloads)} failed")
    
    if failed_downloads:
        logger.info(f"Failed downloads: {failed_downloads[:10]}{'...' if len(failed_downloads) > 10 else ''}")
    
    return downloaded_paths

def load_image(image_path: str, target_size: tuple = (224, 224)) -> Optional[np.ndarray]:
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (height, width)
        
    Returns:
        Preprocessed image array or None if loading fails
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            img = img.resize(target_size)
            
            # Convert to array and normalize
            img_array = np.array(img) / 255.0
            
            return img_array
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {str(e)}")
        return None

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        SMAPE score
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Avoid division by zero
    mask = denominator != 0
    smape = np.zeros_like(numerator)
    smape[mask] = numerator[mask] / denominator[mask]
    
    return np.mean(smape) * 100

def extract_item_pack_quantity(text: str) -> float:
    """
    Extract Item Pack Quantity (IPQ) from catalog content.
    
    Args:
        text: Catalog content text
        
    Returns:
        Extracted quantity as float
    """
    try:
        import re
        
        # Look for "Value: X.X" pattern
        value_match = re.search(r'Value:\s*(\d+\.?\d*)', text, re.IGNORECASE)
        if value_match:
            return float(value_match.group(1))
        
        # Look for quantity patterns in text
        qty_patterns = [
            r'(\d+\.?\d*)\s*(pack|count|oz|ounce|fl\s*oz|pound|lb|gram|g|kg)',
            r'pack\s*of\s*(\d+)',
            r'(\d+)\s*-\s*pack',
        ]
        
        for pattern in qty_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        
        return 1.0  # Default quantity
        
    except:
        return 1.0

def clean_text(text: str) -> str:
    """
    Clean and preprocess text data.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    if pd.isna(text):
        return ""
    
    import re
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_text_features(text: str) -> dict:
    """
    Extract various features from text content.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary of extracted features
    """
    if pd.isna(text):
        text = ""
    
    features = {}
    
    # Basic text statistics
    features['text_length'] = len(str(text))
    features['word_count'] = len(str(text).split())
    features['char_count'] = len(str(text))
    features['sentence_count'] = len(str(text).split('.'))
    
    # Product description indicators
    text_lower = str(text).lower()
    features['has_organic'] = 1 if 'organic' in text_lower else 0
    features['has_natural'] = 1 if 'natural' in text_lower else 0
    features['has_premium'] = 1 if 'premium' in text_lower else 0
    features['has_gourmet'] = 1 if 'gourmet' in text_lower else 0
    features['has_gluten_free'] = 1 if 'gluten free' in text_lower else 0
    features['has_vegan'] = 1 if 'vegan' in text_lower else 0
    features['has_kosher'] = 1 if 'kosher' in text_lower else 0
    features['has_pack'] = 1 if 'pack' in text_lower else 0
    
    # Extract quantity
    features['quantity'] = extract_item_pack_quantity(text)
    
    return features

def save_predictions(sample_ids: List, predictions: List, filename: str = "test_out.csv"):
    """
    Save predictions in the required format.
    
    Args:
        sample_ids: List of sample IDs
        predictions: List of predicted prices
        filename: Output filename
    """
    df = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })
    
    df.to_csv(filename, index=False)
    logger.info(f"Predictions saved to {filename}")