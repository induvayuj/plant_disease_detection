"""
Configuration file for Plant Disease Detection CNN
Contains all hyperparameters and paths
"""

import os

# Dataset Configuration
DATASET_NAME = "plantvillage-dataset"
DATASET_PATH = os.path.join(os.path.expanduser("~"), "plant_disease_data")
NUM_CLASSES = 38  # PlantVillage has 38 disease classes
NUM_CROPS = 14    # Across 14 crop species
RANDOM_SEED = 42

# Image Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
BATCH_SIZE = 32
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2

# Data Augmentation
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Model Configuration
INITIAL_FILTERS = 32
FILTER_MULTIPLIER = 2
DROPOUT_RATE = 0.5
L2_REGULARIZATION = 0.0001

# Training Configuration
EPOCHS = 100
INITIAL_LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# Paths
MODEL_SAVE_PATH = os.path.join(os.path.expanduser("~"), "models")
MODEL_NAME = "plant_disease_detection_model.h5"
BEST_MODEL_NAME = "plant_disease_detection_best.h5"
MODEL_WEIGHTS_PATH = os.path.join(MODEL_SAVE_PATH, BEST_MODEL_NAME)
HISTORY_PATH = os.path.join(MODEL_SAVE_PATH, "training_history.pkl")
LOGS_PATH = os.path.join(os.path.expanduser("~"), "logs")

# Class Names (PlantVillage Dataset - 38 classes)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca', 'Grape___Leaf_blight', 'Grape___healthy',
    'Orange___Haunglongbing', 'Orange___healthy',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___healthy'
]

# Ensure CLASS_NAMES has correct length
assert len(CLASS_NAMES) == NUM_CLASSES, f"CLASS_NAMES length ({len(CLASS_NAMES)}) != NUM_CLASSES ({NUM_CLASSES})"

# Create necessary directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)
