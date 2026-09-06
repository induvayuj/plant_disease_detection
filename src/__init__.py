"""
Plant Disease Detection using CNN
A TensorFlow/Keras implementation for automated plant disease classification
"""

from .config import *
from .model import build_cnn_model, compile_model
from .data import load_images_from_directory, prepare_data_generators
from .train import train_model, get_callbacks
from .evaluate import evaluate_model, plot_training_history
from .predict import PlantDiseasePredictor, predict_disease

__version__ = '1.0.0'
__author__ = 'Plant Disease Detection Team'
__description__ = 'Automated plant disease detection using Convolutional Neural Networks'

__all__ = [
    'build_cnn_model',
    'compile_model',
    'train_model',
    'evaluate_model',
    'predict_disease',
    'PlantDiseasePredictor',
]
