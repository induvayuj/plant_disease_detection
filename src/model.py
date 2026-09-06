"""
CNN Model Architecture for Plant Disease Detection
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import logging

from config import (
    NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, CHANNELS,
    INITIAL_FILTERS, FILTER_MULTIPLIER, DROPOUT_RATE, L2_REGULARIZATION
)

logger = logging.getLogger(__name__)


def build_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)):
    """
    Build a custom CNN model for plant disease detection
    
    Architecture:
    - 3 Convolutional blocks with MaxPooling
    - BatchNormalization for stability
    - Dropout for regularization
    - Dense layers with Dropout
    - Softmax output for 38 classes
    """
    
    model = models.Sequential([
        # Block 1
        layers.Input(shape=input_shape),
        layers.Conv2D(
            INITIAL_FILTERS, (3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
            name='conv1_1'
        ),
        layers.BatchNormalization(),
        layers.Conv2D(
            INITIAL_FILTERS, (3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
            name='conv1_2'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(DROPOUT_RATE),
        
        # Block 2
        layers.Conv2D(
            INITIAL_FILTERS * FILTER_MULTIPLIER, (3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
            name='conv2_1'
        ),
        layers.BatchNormalization(),
        layers.Conv2D(
            INITIAL_FILTERS * FILTER_MULTIPLIER, (3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
            name='conv2_2'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(DROPOUT_RATE),
        
        # Block 3
        layers.Conv2D(
            INITIAL_FILTERS * (FILTER_MULTIPLIER ** 2), (3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
            name='conv3_1'
        ),
        layers.BatchNormalization(),
        layers.Conv2D(
            INITIAL_FILTERS * (FILTER_MULTIPLIER ** 2), (3, 3),
            activation='relu',
            padding='same',
            kernel_regularizer=regularizers.l2(L2_REGULARIZATION),
            name='conv3_2'
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(DROPOUT_RATE),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(L2_REGULARIZATION)),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        
        # Output layer
        layers.Dense(NUM_CLASSES, activation='softmax', name='output')
    ], name='PlantDiseaseDetectionCNN')
    
    logger.info(f"Model built successfully with {model.count_params():,} parameters")
    return model


def build_transfer_learning_model(model_name='MobileNetV2'):
    """
    Build a transfer learning model using pre-trained weights
    Options: 'MobileNetV2', 'ResNet50', 'EfficientNetB0'
    """
    
    base_models = {
        'MobileNetV2': tf.keras.applications.MobileNetV2,
        'ResNet50': tf.keras.applications.ResNet50,
        'EfficientNetB0': tf.keras.applications.EfficientNetB0,
    }
    
    if model_name not in base_models:
        logger.error(f"Model {model_name} not available. Choose from {list(base_models.keys())}")
        return None
    
    # Load pre-trained base model
    base_model = base_models[model_name](
        input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom top layers
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        tf.keras.applications.mobilenet_v2.preprocess_input if model_name == 'MobileNetV2' else layers.Lambda(lambda x: x),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name=f'{model_name}_PlantDisease')
    
    logger.info(f"Transfer learning model ({model_name}) built successfully")
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer and loss function
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    
    logger.info("Model compiled successfully")
    return model


def get_model_summary(model):
    """Print model architecture summary"""
    model.summary()
