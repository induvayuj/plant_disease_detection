# Plant Disease Detection using CNN

A deep learning-based solution for automated plant disease detection and classification using Convolutional Neural Networks (CNNs). This project uses the PlantVillage dataset containing 38 disease classes across 14 different crop species.

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline for:
- **Data Loading & Preprocessing**: Automated dataset download and image normalization
- **Model Architecture**: Custom CNN with batch normalization and dropout regularization
- **Training**: With callbacks for early stopping, learning rate reduction, and model checkpointing
- **Evaluation**: Comprehensive metrics including classification reports and confusion matrices
- **Inference**: Real-time disease prediction on single images or batch processing

## 📊 Dataset

**PlantVillage Dataset**
- **Total Classes**: 38 disease categories
- **Crop Species**: 14 (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)
- **Image Size**: Standardized to 224×224 pixels
- **Format**: JPEG images from Kaggle

### Class Distribution
```
Apple: Apple scab, Black rot, Cedar apple rust, Healthy
Tomato: Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, 
        Spider mites, Target spot, TYLCV, Healthy
Potato: Early blight, Late blight, Healthy
... and more
```

## 🏗️ Project Structure

```
plant_disease_detection/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration & hyperparameters
│   ├── data.py                  # Data loading & preprocessing
│   ├── model.py                 # Model architecture
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Evaluation metrics & plotting
│   └── predict.py               # Inference & prediction
├── notebooks/
│   └── plant_disease_detection.ipynb  # Full Jupyter notebook
├── main.py                      # Entry point script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── LICENSE                      # MIT License
└── .gitignore                   # Git ignore file
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/plant_disease_detection.git
cd plant_disease_detection
```

### 2. Create Virtual Environment
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n plant_disease python=3.9
conda activate plant_disease
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Kaggle API
```bash
# Download kaggle.json from https://www.kaggle.com/settings/account
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 5. Train Model
```bash
# Download dataset and train
python main.py train --epochs 50 --download

# Train without downloading (if dataset exists)
python main.py train --epochs 50
```

### 6. Make Predictions
```bash
# Single image
python main.py predict --image path/to/plant_image.jpg

# Directory of images
python main.py predict --image path/to/images/ --top-k 5

# Get top-5 predictions instead of top-3 (default)
python main.py predict --image plant.jpg --top-k 5
```

### 7. Evaluate Model
```bash
python main.py evaluate
```

## 📋 Detailed Usage

### Training

**Basic Training**
```bash
python main.py train
```

**With Custom Parameters**
```bash
python main.py train --epochs 100 --download
```

**From Python**
```python
from src.model import build_cnn_model, compile_model
from src.train import train_model
from src.data import load_images_from_directory, split_data, normalize_images

# Build model
model = build_cnn_model()
model = compile_model(model, learning_rate=0.001)

# Load and prepare data
images, labels = load_images_from_directory('path/to/data')
images = normalize_images(images)
(X_train, X_val, X_test), (y_train, y_val, y_test) = split_data(images, labels)

# Train
history = train_model(model, (X_train, y_train), (X_val, y_val), epochs=50)
```

### Prediction

**Single Image Prediction**
```python
from src.predict import PlantDiseasePredictor

predictor = PlantDiseasePredictor()
results = predictor.predict_single_image('path/to/image.jpg', top_k=3)

print(f"Disease: {results['primary_prediction']}")
print(f"Confidence: {results['primary_confidence']:.4f}")

for pred in results['top_predictions']:
    print(f"  {pred['class']}: {pred['percentage']:.2f}%")
```

**Batch Prediction**
```python
# Predict on multiple images
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predictor.predict_batch(image_paths)

# Predict directory
results = predictor.predict_directory('path/to/images/')
```

**Get Disease Information**
```python
from src.predict import get_disease_info, print_disease_info

info = get_disease_info('Tomato___Early_blight')
print(f"Crop: {info['crop']}")
print(f"Severity: {info['severity']}")
print(f"Treatment: {info['treatment']}")

# Or print formatted
print_disease_info('Tomato___Early_blight')
```

## 🧠 Model Architecture

### Custom CNN Model
```
Input (224×224×3)
↓
Conv2D (32 filters) + BatchNorm + ReLU
Conv2D (32 filters) + BatchNorm + ReLU
MaxPooling (2×2)
Dropout (0.5)
↓
Conv2D (64 filters) + BatchNorm + ReLU
Conv2D (64 filters) + BatchNorm + ReLU
MaxPooling (2×2)
Dropout (0.5)
↓
Conv2D (128 filters) + BatchNorm + ReLU
Conv2D (128 filters) + BatchNorm + ReLU
MaxPooling (2×2)
Dropout (0.5)
↓
GlobalAveragePooling2D
↓
Dense (256) + BatchNorm + Dropout (0.5)
Dense (128) + BatchNorm + Dropout (0.5)
Dense (38) + Softmax
↓
Output (38 disease classes)
```

### Key Features
- **Batch Normalization**: Stabilizes training and accelerates convergence
- **Dropout Regularization**: Prevents overfitting
- **L2 Regularization**: Reduces model complexity
- **Early Stopping**: Stops training when validation loss plateaus
- **Learning Rate Scheduling**: Reduces LR when validation loss plateaus
- **Class Weighting**: Handles imbalanced dataset

## 📊 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 224×224 | Standardized input dimensions |
| Batch Size | 32 | Training batch size |
| Initial LR | 0.001 | Adam optimizer learning rate |
| Epochs | 100 | Maximum training epochs |
| Dropout Rate | 0.5 | Regularization dropout |
| L2 Reg | 0.0001 | L2 regularization coefficient |
| Early Stop Patience | 15 | Epochs without improvement before stopping |
| LR Reduce Patience | 5 | Epochs without improvement before reducing LR |
| LR Reduce Factor | 0.5 | Factor to multiply LR by |

## 📈 Data Augmentation

Applied to training data only:
- Random rotation (±20°)
- Width/height shift (±20%)
- Shear transformation (±20%)
- Zoom (±20%)
- Horizontal flip
- Fill mode: nearest

## 📝 Results

Expected model performance on PlantVillage dataset:
- **Training Accuracy**: 95%+
- **Validation Accuracy**: 92%+
- **Test Accuracy**: 90%+
- **Top-5 Accuracy**: 97%+

Results vary based on:
- Model variant (custom CNN vs transfer learning)
- Training epochs and dataset size
- Hyperparameter tuning
- Data augmentation strategy

## 🔧 Configuration

Edit `src/config.py` to customize:
```python
# Model
INITIAL_FILTERS = 32          # Base number of filters
DROPOUT_RATE = 0.5            # Dropout probability
L2_REGULARIZATION = 0.0001    # L2 penalty

# Training
EPOCHS = 100                  # Maximum epochs
INITIAL_LEARNING_RATE = 0.001 # Adam LR
EARLY_STOPPING_PATIENCE = 15  # Early stop patience

# Data
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
TEST_SPLIT = 0.2
```

## 🛠️ Transfer Learning

Use pre-trained models for faster training:

```python
from src.model import build_transfer_learning_model, compile_model

# Options: 'MobileNetV2', 'ResNet50', 'EfficientNetB0'
model = build_transfer_learning_model('MobileNetV2')
model = compile_model(model, learning_rate=0.0001)

# Train with lower learning rate
history = train_model(model, (X_train, y_train), (X_val, y_val))
```

## 📚 Jupyter Notebook

Complete walkthrough notebook available in `notebooks/plant_disease_detection.ipynb`:
- Dataset exploration & visualization
- Step-by-step preprocessing
- Model training with live metrics
- Comprehensive evaluation
- Interactive prediction interface

## 🎓 Learning Resources

### Concepts Covered
- Convolutional Neural Networks (CNNs)
- Transfer Learning
- Data Augmentation
- Batch Normalization
- Regularization Techniques
- Model Evaluation Metrics
- Hyperparameter Tuning

### References
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- [CNN Image Classification](https://cs231n.github.io/)

## 🐛 Troubleshooting

### "Module not found" Error
```bash
# Ensure venv is activated and requirements installed
source venv/bin/activate
pip install -r requirements.txt
```

### GPU Not Detected
```bash
# Check TensorFlow GPU setup
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# For CUDA/cuDNN issues, reinstall
pip install tensorflow[and-cuda]
```

### Out of Memory (OOM)
- Reduce batch size in `config.py`: `BATCH_SIZE = 16`
- Use mixed precision training: See `train.py` for `train_with_mixed_precision()`
- Use transfer learning instead of custom CNN

### Kaggle Dataset Download Fails
```bash
# Manual download from Kaggle
# 1. Go to https://www.kaggle.com/emmarex/plantdisease
# 2. Download plantdisease.zip
# 3. Extract to ~/plant_disease_data/
```

## 📊 Evaluation Metrics

The project generates:
- **Confusion Matrix**: Visual representation of predictions vs ground truth
- **Classification Report**: Precision, recall, F1-score per class
- **Training Curves**: Accuracy and loss over epochs
- **Top-K Accuracy**: Performance for top-1, top-3, top-5 predictions
- **Sample Predictions**: Visual examples with confidence scores

## 🔐 Performance Optimization

### Training Speed
- Use GPU: Automatic if CUDA available
- Mixed precision training: 1.5-2x speedup
- Batch size tuning: Larger = faster (if memory allows)

### Inference Speed
- Model quantization: Convert to TFLite/ONNX
- Batch prediction: Process multiple images together
- Model optimization: Prune or compress layers

## 📦 Model Deployment

### Save Trained Model
```python
model.save('plant_disease_model.h5')
```

### Load for Inference
```python
import tensorflow as tf
model = tf.keras.models.load_model('plant_disease_model.h5')
```

### Export Formats
```python
# SavedModel format (recommended)
model.export('plant_disease_savedmodel')

# TFLite for mobile
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional crop species
- Real-world image testing
- Mobile app integration
- Performance optimization
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👤 Author

Plant Disease Detection Team

## 🙏 Acknowledgments

- PlantVillage Dataset contributors
- TensorFlow/Keras communities
- Kaggle platform for dataset hosting

## 📞 Support

For issues, questions, or suggestions:
- Open GitHub issue
- Check troubleshooting section
- Review Jupyter notebook examples

## 📚 Citation

If you use this project in research, please cite:

```bibtex
@software{plant_disease_2024,
  title = {Plant Disease Detection using CNN},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/plant_disease_detection}
}
```

---

**Last Updated**: 2024
**Version**: 1.0.0
**Status**: Production Ready ✅
