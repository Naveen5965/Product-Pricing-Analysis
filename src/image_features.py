"""
Image feature extraction for the Amazon ML Product Pricing Challenge
"""
import numpy as np
import pandas as pd
from PIL import Image
import os
from typing import List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler
import cv2

logger = logging.getLogger(__name__)

class ImageFeatureExtractor:
    """Extract features from product images."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize image feature extractor.
        
        Args:
            target_size: Target size for image resizing (height, width)
        """
        self.target_size = target_size
        self.scaler = StandardScaler()
        self.fitted = False
        
    def load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array or None if loading fails
        """
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return None
                
            # Load image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image
                img = img.resize(self.target_size)
                
                # Convert to numpy array
                img_array = np.array(img)
                
                return img_array
                
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            return None
    
    def extract_color_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color-based features from image.
        
        Args:
            image: Image array (H, W, 3)
            
        Returns:
            Color features array
        """
        features = []
        
        # Convert to different color spaces
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # RGB channel statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image[:, :, i].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.median(channel_data)
            ])
        
        # HSV channel statistics
        for i, channel in enumerate(['H', 'S', 'V']):
            channel_data = image_hsv[:, :, i].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data)
            ])
        
        # LAB channel statistics
        for i, channel in enumerate(['L', 'A', 'B']):
            channel_data = image_lab[:, :, i].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data)
            ])
        
        # Overall brightness and contrast
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features.extend([
            np.mean(gray),  # Average brightness
            np.std(gray)    # Contrast
        ])
        
        return np.array(features)
    
    def extract_texture_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract texture features using statistical methods.
        
        Args:
            image: Image array (H, W, 3)
            
        Returns:
            Texture features array
        """
        features = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.max(magnitude)
        ])
        
        # Edge density (using Canny edge detector)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Local Binary Pattern-like features (simplified)
        # Calculate variance in local neighborhoods
        kernel_sizes = [3, 5, 7]
        for ksize in kernel_sizes:
            kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
            mean_filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            variance = (gray.astype(np.float32) - mean_filtered) ** 2
            features.append(np.mean(variance))
        
        return np.array(features)
    
    def extract_shape_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract shape-based features.
        
        Args:
            image: Image array (H, W, 3)
            
        Returns:
            Shape features array
        """
        features = []
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Simple thresholding to find objects
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate area and perimeter
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Calculate shape descriptors
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Extent (object area / bounding box area)
            extent = area / (w * h) if (w * h) > 0 else 0
            
            features.extend([area, perimeter, circularity, aspect_ratio, extent])
        else:
            features.extend([0, 0, 0, 1, 0])
        
        # Number of contours (complexity indicator)
        features.append(len(contours))
        
        return np.array(features)
    
    def extract_histogram_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract histogram-based features.
        
        Args:
            image: Image array (H, W, 3)
            
        Returns:
            Histogram features array
        """
        features = []
        
        # RGB histograms
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [32], [0, 256])
            hist = hist.flatten() / np.sum(hist)  # Normalize
            features.extend(hist)
        
        # HSV histograms
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for i in range(3):
            hist = cv2.calcHist([image_hsv], [i], None, [16], [0, 256])
            hist = hist.flatten() / np.sum(hist)  # Normalize
            features.extend(hist)
        
        return np.array(features)
    
    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract all image features.
        
        Args:
            image: Image array (H, W, 3)
            
        Returns:
            Combined feature array
        """
        color_features = self.extract_color_features(image)
        texture_features = self.extract_texture_features(image)
        shape_features = self.extract_shape_features(image)
        hist_features = self.extract_histogram_features(image)
        
        # Combine all features
        all_features = np.concatenate([
            color_features,
            texture_features,
            shape_features,
            hist_features
        ])
        
        return all_features
    
    def process_image_paths(self, image_paths: List[str], sample_ids: List) -> np.ndarray:
        """
        Process multiple images and extract features.
        
        Args:
            image_paths: List of image file paths
            sample_ids: List of corresponding sample IDs
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        all_features = []
        successful_indices = []
        
        logger.info(f"Processing {len(image_paths)} images...")
        
        for idx, (image_path, sample_id) in enumerate(zip(image_paths, sample_ids)):
            # Try to load image
            image = self.load_and_preprocess_image(image_path)
            
            if image is not None:
                try:
                    features = self.extract_all_features(image)
                    all_features.append(features)
                    successful_indices.append(idx)
                except Exception as e:
                    logger.error(f"Feature extraction failed for {sample_id}: {str(e)}")
            else:
                logger.warning(f"Could not process image for sample {sample_id}")
        
        if not all_features:
            logger.error("No images could be processed!")
            # Return dummy features
            dummy_features = np.zeros((len(image_paths), 200))  # Estimated feature count
            return dummy_features
        
        # Convert to numpy array
        feature_matrix = np.array(all_features)
        
        # For missing images, use mean features
        if len(successful_indices) < len(image_paths):
            mean_features = np.mean(feature_matrix, axis=0)
            full_feature_matrix = np.zeros((len(image_paths), feature_matrix.shape[1]))
            
            # Fill successful features
            for i, idx in enumerate(successful_indices):
                full_feature_matrix[idx] = feature_matrix[i]
            
            # Fill missing features with mean
            missing_indices = set(range(len(image_paths))) - set(successful_indices)
            for idx in missing_indices:
                full_feature_matrix[idx] = mean_features
            
            feature_matrix = full_feature_matrix
        
        logger.info(f"Extracted {feature_matrix.shape[1]} image features for {feature_matrix.shape[0]} samples")
        return feature_matrix
    
    def fit(self, image_paths: List[str], sample_ids: List) -> 'ImageFeatureExtractor':
        """
        Fit the image feature extractor.
        
        Args:
            image_paths: List of image paths for training
            sample_ids: List of corresponding sample IDs
            
        Returns:
            Self
        """
        # Extract features
        features = self.process_image_paths(image_paths, sample_ids)
        
        # Fit scaler
        self.scaler.fit(features)
        self.fitted = True
        
        return self
    
    def transform(self, image_paths: List[str], sample_ids: List) -> np.ndarray:
        """
        Transform images to features.
        
        Args:
            image_paths: List of image paths
            sample_ids: List of corresponding sample IDs
            
        Returns:
            Scaled feature matrix
        """
        if not self.fitted:
            raise ValueError("Must fit before transform")
        
        # Extract features
        features = self.process_image_paths(image_paths, sample_ids)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def fit_transform(self, image_paths: List[str], sample_ids: List) -> np.ndarray:
        """
        Fit and transform images.
        
        Args:
            image_paths: List of image paths
            sample_ids: List of corresponding sample IDs
            
        Returns:
            Scaled feature matrix
        """
        return self.fit(image_paths, sample_ids).transform(image_paths, sample_ids)