# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites
- Python 3.8+
- pip or conda
- 8GB RAM (16GB recommended)
- GPU optional but recommended

### Step 1: Setup Environment
```bash
# Clone/Extract project
cd plant_disease_detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Setup Kaggle API (One-time)
```bash
# Download from https://www.kaggle.com/settings/account
# Click "Create New API Token"

mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json  # Linux/Mac only
```

### Step 3: Train Model
```bash
# Download dataset and train (first time takes 1-2 hours)
python main.py train --epochs 50 --download

# Or use Jupyter notebook for interactive training
jupyter notebook notebooks/plant_disease_detection.ipynb
```

### Step 4: Make Predictions
```bash
# Single image prediction
python main.py predict --image path/to/image.jpg

# Directory prediction
python main.py predict --image path/to/images/ --top-k 5
```

## 📝 Usage Examples

### Training
```bash
# With 100 epochs
python main.py train --epochs 100 --download

# Without downloading (if dataset already exists)
python main.py train --epochs 50
```

### Prediction
```bash
# Top-3 predictions (default)
python main.py predict --image plant.jpg

# Top-5 predictions
python main.py predict --image plant.jpg --top-k 5

# Batch process directory
python main.py predict --image ./plant_images/
```

### Evaluation
```bash
python main.py evaluate
```

## 🐍 Python API Usage

```python
from src.model import build_cnn_model, compile_model
from src.train import train_model
from src.data import load_images_from_directory, split_data, normalize_images
from src.predict import PlantDiseasePredictor

# Make predictions
predictor = PlantDiseasePredictor()
results = predictor.predict_single_image('plant.jpg')

print(f"Disease: {results['primary_prediction']}")
print(f"Confidence: {results['primary_confidence']:.2%}")
```

## 📊 Expected Performance

- **Training**: 95%+ accuracy
- **Validation**: 92%+ accuracy  
- **Test**: 90%+ accuracy
- **Top-5**: 97%+ accuracy

## 🔧 Troubleshooting

### "Module not found" Error
```bash
# Verify you're in the venv
source venv/bin/activate
pip install -r requirements.txt
```

### GPU Not Working
```bash
# Check GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, reinstall with CUDA support
pip install tensorflow[and-cuda]
```

### Out of Memory
Edit `src/config.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
```

### Dataset Download Fails
1. Go to https://www.kaggle.com/emmarex/plantdisease
2. Download plantdisease.zip manually
3. Extract to `~/plant_disease_data/`

## 📁 Directory Structure After First Run

```
plant_disease_detection/
├── src/                          # Source code
├── notebooks/                    # Jupyter notebooks
├── plant_disease_data/           # Dataset (auto-downloaded)
├── models/                       # Saved models
├── logs/                         # Training logs
└── results/                      # Evaluation results
```

## 🎓 Next Steps

1. **Explore Data**: Run EDA in Jupyter notebook
2. **Train Model**: `python main.py train --download`
3. **Evaluate**: `python main.py evaluate`
4. **Predict**: `python main.py predict --image sample.jpg`
5. **Deploy**: See README.md for deployment options

## ⏱️ Typical Training Time

| Hardware | Time for 50 Epochs |
|----------|-------------------|
| CPU | 8-12 hours |
| GPU (NVIDIA) | 30-60 minutes |
| Google Colab | 45-90 minutes |

## 📚 Learn More

- Full documentation: See `README.md`
- Jupyter notebook: `notebooks/plant_disease_detection.ipynb`
- Configuration: Edit `src/config.py`

## 🤔 Common Questions

**Q: How long does training take?**
A: 30-60 minutes with GPU, 8-12 hours with CPU

**Q: Can I use a pre-trained model?**
A: Yes! See `src/model.py` for transfer learning options

**Q: How do I deploy this?**
A: See deployment section in README.md

**Q: Can I train on custom data?**
A: Yes! Organize images in `class/disease_name/image.jpg` format

## 🎯 Quick Command Reference

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Training
python main.py train --download --epochs 50

# Prediction
python main.py predict --image test.jpg

# Jupyter
jupyter notebook notebooks/plant_disease_detection.ipynb

# Git
git init
git add .
git commit -m "Initial commit"
git push origin main
```

---

**Need help?** Check README.md or open an issue on GitHub!
