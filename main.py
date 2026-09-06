"""
Main script for Plant Disease Detection
Handles training, evaluation, and inference
"""

import os
import sys
import argparse
import logging
import pickle
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import tensorflow as tf
from src.config import (
    DATASET_PATH, MODEL_SAVE_PATH, HISTORY_PATH,
    INITIAL_LEARNING_RATE, EPOCHS, CLASS_NAMES
)
from src.model import build_cnn_model, compile_model
from src.data import (
    download_kaggle_dataset, load_images_from_directory,
    normalize_images, split_data, get_class_weights, create_tf_dataset
)
from src.train import train_model, save_training_summary
from src.evaluate import (
    evaluate_model, plot_training_history, plot_confusion_matrix,
    print_classification_report, plot_top_predictions, plot_top_k_accuracy
)
from src.predict import PlantDiseasePredictor, print_prediction_results, print_disease_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('plant_disease_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_gpu():
    """Setup GPU for training"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.warning(f"GPU setup error: {e}")
    else:
        logger.info("No GPU found, will use CPU")


def train_pipeline(args):
    """Complete training pipeline"""
    logger.info("Starting training pipeline...")
    
    setup_gpu()
    
    # Download dataset if needed
    if not os.path.exists(DATASET_PATH):
        logger.info(f"Dataset not found at {DATASET_PATH}")
        if args.download:
            logger.info("Downloading dataset from Kaggle...")
            if not download_kaggle_dataset():
                logger.error("Failed to download dataset. Please download manually or check Kaggle credentials.")
                return
        else:
            logger.error("Dataset not found. Use --download flag to download from Kaggle.")
            return
    
    # Find dataset directory
    dataset_dir = DATASET_PATH
    if os.path.exists(os.path.join(DATASET_PATH, 'PlantVillage')):
        dataset_dir = os.path.join(DATASET_PATH, 'PlantVillage')
    elif os.path.exists(os.path.join(DATASET_PATH, 'plant_disease')):
        dataset_dir = os.path.join(DATASET_PATH, 'plant_disease')
    
    logger.info(f"Using dataset from: {dataset_dir}")
    
    # Load data
    logger.info("Loading images...")
    images, labels = load_images_from_directory(dataset_dir)
    
    if images is None:
        logger.error("Failed to load images")
        return
    
    logger.info(f"Loaded {len(images)} images from {len(np.unique(labels))} classes")
    
    # Preprocess
    logger.info("Normalizing images...")
    images = normalize_images(images)
    
    # Convert labels to one-hot
    labels_onehot = tf.keras.utils.to_categorical(labels, len(CLASS_NAMES))
    
    # Split data
    logger.info("Splitting data...")
    (X_train, X_val, X_test), (y_train, y_val, y_test) = split_data(images, labels_onehot)
    
    # Get class weights for imbalanced data
    class_weights = get_class_weights(labels)
    
    # Build model
    logger.info("Building model...")
    model = build_cnn_model()
    model = compile_model(model, INITIAL_LEARNING_RATE)
    model.summary()
    
    # Train model
    logger.info("Training model...")
    history = train_model(
        model, 
        (X_train, y_train), 
        (X_val, y_val),
        epochs=args.epochs if args.epochs else EPOCHS,
        class_weights=class_weights,
        experiment_name='plant_disease_detection'
    )
    
    # Evaluate
    logger.info("Evaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    # Save plots
    logger.info("Generating evaluation plots...")
    plots_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load history from pickle
    with open(HISTORY_PATH, 'rb') as f:
        history_dict = pickle.load(f)
    
    plot_training_history(history_dict, os.path.join(plots_dir, 'training_history.png'))
    plot_confusion_matrix(results['y_true'], results['y_pred'], 
                         os.path.join(plots_dir, 'confusion_matrix.png'))
    plot_top_k_accuracy(model, X_test, y_test, 
                        save_path=os.path.join(plots_dir, 'top_k_accuracy.png'))
    plot_top_predictions(model, X_test, y_test, 
                        save_path=os.path.join(plots_dir, 'sample_predictions.png'))
    
    # Print reports
    print_classification_report(results['y_true'], results['y_pred'],
                               os.path.join(plots_dir, 'classification_report.txt'))
    
    logger.info(f"Results saved to {plots_dir}")
    logger.info("Training completed successfully!")


def predict_pipeline(args):
    """Inference pipeline"""
    logger.info(f"Starting inference on {args.image_path}...")
    
    try:
        predictor = PlantDiseasePredictor()
        
        if os.path.isdir(args.image_path):
            logger.info(f"Predicting on directory: {args.image_path}")
            results = predictor.predict_directory(args.image_path, top_k=args.top_k)
        else:
            logger.info(f"Predicting on single image: {args.image_path}")
            results = predictor.predict_single_image(args.image_path, top_k=args.top_k)
        
        # Print results
        print_prediction_results(results, verbose=True)
        
        # Print disease info if single prediction
        if isinstance(results, dict):
            print_disease_info(results['primary_prediction'])
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")


def evaluate_pipeline(args):
    """Evaluation pipeline"""
    logger.info("Starting evaluation...")
    
    # Load test data
    dataset_dir = DATASET_PATH
    if os.path.exists(os.path.join(DATASET_PATH, 'PlantVillage')):
        dataset_dir = os.path.join(DATASET_PATH, 'PlantVillage')
    
    logger.info("Loading test images...")
    images, labels = load_images_from_directory(dataset_dir)
    
    if images is None:
        logger.error("Failed to load images")
        return
    
    images = normalize_images(images)
    labels_onehot = tf.keras.utils.to_categorical(labels, len(CLASS_NAMES))
    
    # Use last 20% as test set
    split_idx = int(len(images) * 0.8)
    X_test = images[split_idx:]
    y_test = labels_onehot[split_idx:]
    
    # Load model
    from src.config import MODEL_WEIGHTS_PATH
    model = tf.keras.models.load_model(MODEL_WEIGHTS_PATH)
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test)
    
    logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logger.info("Evaluation completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Plant Disease Detection using CNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python main.py train --epochs 50 --download
  
  # Predict on single image
  python main.py predict --image path/to/image.jpg
  
  # Predict on directory
  python main.py predict --image path/to/images/
  
  # Evaluate model
  python main.py evaluate
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=EPOCHS,
                             help=f'Number of epochs (default: {EPOCHS})')
    train_parser.add_argument('--download', action='store_true',
                             help='Download dataset from Kaggle')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--image', dest='image_path', required=True,
                               help='Path to image or directory')
    predict_parser.add_argument('--top-k', type=int, default=3,
                               help='Show top-k predictions (default: 3)')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_pipeline(args)
    elif args.command == 'predict':
        predict_pipeline(args)
    elif args.command == 'evaluate':
        evaluate_pipeline(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
