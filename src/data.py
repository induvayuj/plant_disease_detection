"""
Data loading and preprocessing for Plant Disease Detection
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import logging

from config import (
    DATASET_PATH, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, 
    TEST_SPLIT, VAL_SPLIT, AUGMENTATION_CONFIG, RANDOM_SEED, CLASS_NAMES
)

logger = logging.getLogger(__name__)


def download_kaggle_dataset():
    """
    Download PlantVillage dataset from Kaggle using kaggle CLI
    Requires: kaggle credentials configured at ~/.kaggle/kaggle.json
    """
    try:
        import subprocess
        logger.info("Downloading PlantVillage dataset from Kaggle...")
        subprocess.run([
            'kaggle', 'datasets', 'download', '-d', 'emmarex/plantdisease',
            '-p', DATASET_PATH, '--unzip'
        ], check=True)
        logger.info(f"Dataset downloaded to {DATASET_PATH}")
        return True
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return False


def load_images_from_directory(directory_path):
    """
    Load images from directory and labels from folder structure
    Expected structure: /disease_name/image_0001.jpg
    """
    images = []
    labels = []
    
    if not os.path.exists(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return None, None
    
    # Get all class folders
    class_folders = sorted([d for d in os.listdir(directory_path) 
                           if os.path.isdir(os.path.join(directory_path, d))])
    
    logger.info(f"Found {len(class_folders)} disease classes")
    
    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(directory_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        logger.info(f"Loading class {class_idx}: {class_name} ({len(image_files)} images)")
        
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            try:
                image = tf.keras.preprocessing.image.load_img(
                    image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
                )
                image_array = tf.keras.preprocessing.image.img_to_array(image)
                images.append(image_array)
                labels.append(class_idx)
            except Exception as e:
                logger.warning(f"Error loading image {image_path}: {e}")
    
    return np.array(images), np.array(labels)


def normalize_images(images):
    """Normalize images to [0, 1] range"""
    return images / 255.0


def prepare_data_generators():
    """
    Create train, validation, and test data generators with augmentation
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        **AUGMENTATION_CONFIG
    )
    
    # Validation and test data generators (only rescaling, no augmentation)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_test_datagen


def create_data_generators_from_directory(base_path):
    """
    Create data generators directly from directory structure
    Expects: base_path/train/disease_name/ and base_path/validation/disease_name/
    """
    train_datagen, val_test_datagen = prepare_data_generators()
    
    train_path = os.path.join(base_path, 'train')
    val_path = os.path.join(base_path, 'validation')
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_SEED
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_path if os.path.exists(val_path) else train_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_SEED
    )
    
    return train_generator, val_generator


def split_data(images, labels, test_size=TEST_SPLIT, val_size=VAL_SPLIT):
    """
    Split data into train, validation, and test sets
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=RANDOM_SEED, stratify=labels
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=RANDOM_SEED, stratify=y_temp
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return (X_train, X_val, X_test), (y_train, y_val, y_test)


def create_tf_dataset(images, labels, batch_size=BATCH_SIZE, augment=False):
    """
    Create TensorFlow dataset from images and labels
    """
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if augment:
        dataset = dataset.map(
            lambda x, y: (tf.image.random_flip_left_right(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(
            lambda x, y: (tf.image.random_rotation(x, 0.2), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def get_class_weights(labels):
    """Calculate class weights to handle imbalanced dataset"""
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    class_weights = {}
    for class_idx, count in zip(unique_classes, class_counts):
        class_weights[class_idx] = total_samples / (len(unique_classes) * count)
    
    logger.info(f"Class weights: {class_weights}")
    return class_weights
