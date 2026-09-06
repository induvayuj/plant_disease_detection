"""
Inference and Prediction Module for Plant Disease Detection
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import logging
import os
from pathlib import Path

from config import IMG_HEIGHT, IMG_WIDTH, CLASS_NAMES, MODEL_WEIGHTS_PATH

logger = logging.getLogger(__name__)


class PlantDiseasePredictor:
    """
    Class for making predictions on plant disease images
    """
    
    def __init__(self, model_path=MODEL_WEIGHTS_PATH):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained model file (.h5)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        logger.info(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = CLASS_NAMES
        logger.info("Model loaded successfully!")
    
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess image for prediction
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image array and original image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        # Load image
        image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        image_array = img_to_array(image)
        
        # Normalize
        image_normalized = image_array / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch, image
    
    def predict_single_image(self, image_path, top_k=3):
        """
        Predict disease for a single image
        
        Args:
            image_path: Path to image file
            top_k: Return top-k predictions
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        try:
            image_batch, original_image = self.load_and_preprocess_image(image_path)
            
            # Get predictions
            predictions = self.model.predict(image_batch, verbose=0)
            prediction_probs = predictions[0]
            
            # Get top-k predictions
            top_k_indices = np.argsort(prediction_probs)[-top_k:][::-1]
            
            results = {
                'image_path': image_path,
                'top_predictions': []
            }
            
            for idx in top_k_indices:
                results['top_predictions'].append({
                    'class': self.class_names[idx],
                    'confidence': float(prediction_probs[idx]),
                    'percentage': float(prediction_probs[idx] * 100)
                })
            
            # Primary prediction
            primary_idx = np.argmax(prediction_probs)
            results['primary_prediction'] = self.class_names[primary_idx]
            results['primary_confidence'] = float(prediction_probs[primary_idx])
            
            return results
        
        except Exception as e:
            logger.error(f"Error predicting image {image_path}: {e}")
            return None
    
    def predict_batch(self, image_paths, top_k=3):
        """
        Predict diseases for multiple images
        
        Args:
            image_paths: List of image file paths
            top_k: Return top-k predictions per image
        
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict_single_image(image_path, top_k)
            if result:
                results.append(result)
        
        logger.info(f"Batch prediction completed for {len(results)} images")
        return results
    
    def predict_directory(self, directory_path, top_k=3, file_extensions=('.jpg', '.jpeg', '.png')):
        """
        Predict diseases for all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            top_k: Return top-k predictions per image
            file_extensions: File extensions to look for
        
        Returns:
            List of prediction results
        """
        image_paths = []
        
        for ext in file_extensions:
            image_paths.extend(Path(directory_path).glob(f'*{ext}'))
            image_paths.extend(Path(directory_path).glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_paths)} images in {directory_path}")
        
        return self.predict_batch([str(p) for p in image_paths], top_k)


def predict_disease(image_path, model_path=MODEL_WEIGHTS_PATH, top_k=3):
    """
    Convenience function to predict disease from image
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model
        top_k: Return top-k predictions
    
    Returns:
        Dictionary with prediction results
    """
    predictor = PlantDiseasePredictor(model_path)
    return predictor.predict_single_image(image_path, top_k)


def print_prediction_results(results, verbose=True):
    """
    Print prediction results in readable format
    
    Args:
        results: Dictionary or list of prediction results
        verbose: Print detailed information
    """
    if isinstance(results, list):
        for result in results:
            print_prediction_results(result, verbose)
    else:
        print("\n" + "="*50)
        print(f"Image: {results['image_path']}")
        print(f"Primary Prediction: {results['primary_prediction']}")
        print(f"Confidence: {results['primary_confidence']:.4f} ({results['primary_confidence']*100:.2f}%)")
        
        if verbose:
            print("\nTop Predictions:")
            for i, pred in enumerate(results['top_predictions'], 1):
                print(f"  {i}. {pred['class']}: {pred['percentage']:.2f}%")
        print("="*50 + "\n")


def get_disease_info(disease_name):
    """
    Get information about a specific disease
    
    Args:
        disease_name: Name of the disease (from CLASS_NAMES)
    
    Returns:
        Dictionary with disease information
    """
    # Disease information database
    disease_info = {
        'Apple___Apple_scab': {
            'crop': 'Apple',
            'disease': 'Apple Scab',
            'description': 'Fungal disease causing dark spots on leaves and fruit',
            'severity': 'High',
            'treatment': 'Fungicides, pruning infected branches, good sanitation'
        },
        'Apple___Black_rot': {
            'crop': 'Apple',
            'disease': 'Black Rot',
            'description': 'Fungal disease causing dark, sunken lesions',
            'severity': 'High',
            'treatment': 'Remove infected fruit and branches, fungicide application'
        },
        'Tomato___Early_blight': {
            'crop': 'Tomato',
            'disease': 'Early Blight',
            'description': 'Fungal disease causing circular spots on leaves',
            'severity': 'Medium',
            'treatment': 'Fungicides, remove infected leaves, improve air circulation'
        },
        'Tomato___Late_blight': {
            'crop': 'Tomato',
            'disease': 'Late Blight',
            'description': 'Destructive fungal disease, causes water-soaked spots',
            'severity': 'Very High',
            'treatment': 'Fungicides, resistant varieties, avoid overhead watering'
        },
        'Potato___Early_blight': {
            'crop': 'Potato',
            'disease': 'Early Blight',
            'description': 'Fungal disease causing concentric circular spots',
            'severity': 'Medium',
            'treatment': 'Fungicides, crop rotation, remove infected foliage'
        },
        'Potato___Late_blight': {
            'crop': 'Potato',
            'disease': 'Late Blight',
            'description': 'Severe fungal disease, critical for potato crops',
            'severity': 'Very High',
            'treatment': 'Resistant varieties, fungicides, proper storage'
        }
    }
    
    if disease_name in disease_info:
        return disease_info[disease_name]
    else:
        # Extract crop and disease from class name
        parts = disease_name.split('___')
        return {
            'crop': parts[0],
            'disease': parts[1] if len(parts) > 1 else 'Unknown',
            'description': 'Disease information not available',
            'severity': 'Unknown',
            'treatment': 'Consult agricultural extension services'
        }


def print_disease_info(disease_name):
    """
    Print detailed information about a disease
    """
    info = get_disease_info(disease_name)
    print("\n" + "="*50)
    print(f"Disease Information")
    print("="*50)
    print(f"Crop: {info['crop']}")
    print(f"Disease: {info['disease']}")
    print(f"Description: {info['description']}")
    print(f"Severity: {info['severity']}")
    print(f"Treatment: {info['treatment']}")
    print("="*50 + "\n")
