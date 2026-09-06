"""
Training pipeline for Plant Disease Detection Model
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)
import logging

from config import (
    MODEL_SAVE_PATH, BEST_MODEL_NAME, HISTORY_PATH, LOGS_PATH,
    EPOCHS, EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    INITIAL_LEARNING_RATE
)
from model import build_cnn_model, compile_model

logger = logging.getLogger(__name__)


def get_callbacks(experiment_name='plant_disease_detection'):
    """
    Create training callbacks for model monitoring and saving
    """
    callbacks = [
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        
        # Save best model
        ModelCheckpoint(
            os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        
        # TensorBoard for visualization
        TensorBoard(
            log_dir=os.path.join(LOGS_PATH, experiment_name),
            histogram_freq=1,
            write_graph=True
        ),
        
        # CSV logger for training history
        CSVLogger(
            os.path.join(LOGS_PATH, f'{experiment_name}_training.csv')
        )
    ]
    
    return callbacks


def train_model(model, train_data, val_data, epochs=EPOCHS, 
                class_weights=None, experiment_name='plant_disease_detection'):
    """
    Train the model
    
    Args:
        model: Compiled Keras model
        train_data: Training data (X_train, y_train) or data generator
        val_data: Validation data (X_val, y_val) or data generator
        epochs: Number of training epochs
        class_weights: Dictionary of class weights for imbalanced data
        experiment_name: Name for logging and checkpoints
    
    Returns:
        history: Training history object
    """
    
    callbacks = get_callbacks(experiment_name)
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    # Handle both array and generator inputs
    if isinstance(train_data, tuple):
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
    else:
        # Data generator input
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=len(train_data),
            validation_steps=len(val_data)
        )
    
    logger.info("Training completed!")
    
    # Save training history
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    logger.info(f"Training history saved to {HISTORY_PATH}")
    
    return history


def train_with_mixed_precision(model, train_data, val_data, epochs=EPOCHS, 
                                class_weights=None, experiment_name='plant_disease_detection'):
    """
    Train model using mixed precision for faster training on GPUs
    """
    
    # Enable mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    logger.info("Mixed precision training enabled")
    
    callbacks = get_callbacks(experiment_name)
    
    if isinstance(train_data, tuple):
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,  # Can use larger batch with mixed precision
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
    else:
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    
    # Save training history
    with open(HISTORY_PATH, 'wb') as f:
        pickle.dump(history.history, f)
    
    return history


def save_training_summary(history, save_path=None):
    """
    Save training summary statistics
    """
    if save_path is None:
        save_path = os.path.join(LOGS_PATH, 'training_summary.txt')
    
    summary = f"""
    Training Summary
    ================
    
    Final Training Accuracy: {history.history['accuracy'][-1]:.4f}
    Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}
    
    Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}
    Best Training Accuracy: {max(history.history['accuracy']):.4f}
    
    Final Training Loss: {history.history['loss'][-1]:.4f}
    Final Validation Loss: {history.history['val_loss'][-1]:.4f}
    
    Total Epochs: {len(history.history['loss'])}
    """
    
    with open(save_path, 'w') as f:
        f.write(summary)
    
    logger.info(summary)
    return summary
