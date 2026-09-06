# 🌿 Plant Disease Detection

An AI-powered image classification project that detects plant diseases from leaf images using **Deep Learning and Convolutional Neural Networks (CNN)**.

## 📌 Project Overview

Plant diseases can significantly reduce crop production and affect agricultural productivity. This project uses **Computer Vision and Deep Learning** to identify diseases from plant leaf images.

The model analyzes an uploaded leaf image and predicts the corresponding plant disease.

## 🎯 Objectives

* Detect plant diseases automatically from leaf images.
* Use Deep Learning for image classification.
* Reduce the need for manual disease identification.
* Provide a simple foundation for an AI-based agricultural solution.

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **Convolutional Neural Network (CNN)**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Google Colab**

## 📂 Dataset

The model is trained using the **PlantVillage dataset**, a widely used dataset for plant disease classification.

The dataset contains images of healthy and diseased plant leaves belonging to different plant species and disease categories.

## ⚙️ Project Workflow

```text
Leaf Image
     ↓
Image Preprocessing
     ↓
Resize & Normalize Image
     ↓
CNN Model
     ↓
Feature Extraction
     ↓
Disease Classification
     ↓
Predicted Disease
```

## 🧠 Model

A **Convolutional Neural Network (CNN)** is used for image classification.

CNNs are well suited for image-based tasks because they can automatically learn important visual features such as:

* Edges
* Shapes
* Textures
* Patterns
* Disease symptoms

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/induvayuj/plant_disease_detection.git
```

### 2. Navigate to the project folder

```bash
cd plant_disease_detection
```

### 3. Install the required libraries

```bash
pip install tensorflow keras numpy matplotlib seaborn scikit-learn
```

### 4. Run the project

Open the project notebook/code in **Google Colab or Jupyter Notebook** and execute the cells.

## 📊 Results

The trained CNN model can classify plant leaf images into their corresponding disease categories.

Model performance can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## 🔮 Future Improvements

* Develop a web application for real-time disease detection.
* Add more plant species and disease categories.
* Improve model accuracy using transfer learning.
* Deploy the model using Flask or FastAPI.
* Create a mobile application for farmers.
* Provide disease treatment and prevention recommendations.

## 👩‍💻 Author

**Indu Vayu J**

B.Tech – Information Technology

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
