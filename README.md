# 🧠 Brain Tumor Detection using Deep Learning

A CNN-based system for automated brain tumor detection in MRI scans, with **Grad-CAM explainability** so you can see exactly where the model is looking.

![Grad-CAM Explanations](gradcam_explanations.png)

## 🎯 Project Overview

Brain tumors are among the most serious medical conditions requiring early detection. This project uses a custom CNN to classify MRI scans as Tumor or Healthy, and Grad-CAM heatmaps to visualize which brain regions influenced each decision.

### Key Features
- ✅ Binary classification (Tumor / Healthy)
- ✅ Custom CNN — 4 convolutional blocks, 6.8M parameters
- ✅ **Grad-CAM heatmaps** — see what the model actually learned
- ✅ Data augmentation for better generalization
- ✅ Streamlit web app with real-time prediction + heatmap overlay
- ✅ Full evaluation: accuracy, precision, recall, F1, sensitivity, specificity

## 📊 Dataset

- **Source**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — Kaggle
- **Total Images**: 5,712 MRI scans
- **Classes**:
  - Tumor: 4,117 images (glioma, meningioma, pituitary)
  - Healthy: 1,595 images (no tumor)
- **Split**: 70% train · 15% validation · 15% test

![Data Distribution](binary_distribution.png)

## 🏗️ Model Architecture

```
Input (150×150×3)
    ↓
Conv2D (32)  + BatchNorm + MaxPool
    ↓
Conv2D (64)  + BatchNorm + MaxPool
    ↓
Conv2D (128) + BatchNorm + MaxPool
    ↓
Conv2D (256) + BatchNorm + MaxPool   ← Grad-CAM targets this layer
    ↓
Flatten → Dropout(0.5)
    ↓
Dense (256) → Dropout(0.3)
    ↓
Dense (1, sigmoid)
```

**Total Parameters**: 3,602,113 (~13.7 MB)

## 🔥 Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) shows **which parts of the MRI the model used** to make its prediction.

- 🔴 **Red / warm regions** = highest influence on the decision
- 🔵 **Blue / cool regions** = largely ignored by the model

This is critical for medical AI — a model that just says "Tumor" without showing *why* is hard to trust. Grad-CAM makes the decision transparent.

![Grad-CAM Grid](gradcam_explanations.png)

## 📈 Results

| Metric | Value |
|---|---|
| Test Accuracy | **97.78%** |
| Sensitivity (Tumor Detection Rate) | **96.76%** |
| Specificity (Healthy ID Rate) | **100%** |
| Precision (Tumor) | **100%** |
| Precision (Healthy) | **92%** |
| F1-Score (Tumor) | **98%** |
| F1-Score (Healthy) | **96%** |

The model correctly identified **597 out of 617 tumors** (only 20 false negatives) and **all 240 healthy scans** (zero false positives). The perfect specificity (100%) means the model never incorrectly flags a healthy scan as tumor, making it reliable for screening purposes.

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### Training History
![Training History](training_history.png)

## 🚀 Getting Started

### Prerequisites
```
Python 3.8+
TensorFlow 2.x
OpenCV
scikit-learn
matplotlib
Pillow
Streamlit
```

### Installation

```bash
# 1. Clone
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt

# 4. Dataset
# Download from https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
# Extract into dataset/ at the project root
```

### Usage

#### Train + Evaluate
```bash
python detection.py
```
Trains the model, runs full evaluation, generates all plots (including the Grad-CAM grid), and saves the best model checkpoint.

#### Predict a Single Image
```python
from keras.models import load_model
from detection import predict_image

model = load_model('best_brain_tumor_model.h5')
result, confidence = predict_image(model, 'path/to/mri_scan.jpg')
# Saves a 3-panel figure: Original | Heatmap | Overlay → prediction_result.png
```

#### Run the Web App
```bash
streamlit run app.py
```
Upload any MRI scan → get a prediction + Grad-CAM heatmap overlay in the browser.

## 📁 Project Structure

```
brain-tumor-detection/
├── dataset/
│   └── Training/
│       ├── glioma/
│       ├── meningioma/
│       ├── pituitary/
│       └── notumor/
├── detection.py                  # Training, evaluation, Grad-CAM
├── app.py                        # Streamlit web app
├── requirements.txt              # Python dependencies
├── .gitignore
├── README.md
├── best_brain_tumor_model.h5     # Best checkpoint (generated after training)
├── model_config.json             # Optimized threshold config (generated)
└── *.png                         # Generated plots (binary_distribution, confusion_matrix, etc.)
```

## 🔮 Future Improvements

- [ ] Transfer learning (ResNet50 / EfficientNet) for higher accuracy
- [ ] Multi-class classification (glioma / meningioma / pituitary / healthy)
- [ ] TensorFlow Lite export for mobile deployment
- [ ] Confidence thresholds for uncertain predictions

## 🛠️ Technologies

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Explainability | Grad-CAM (GradientTape) |
| Computer Vision | OpenCV |
| Web App | Streamlit |
| Evaluation | scikit-learn |
| Data | NumPy, Pandas, Matplotlib |

## ⚠️ Disclaimer

This is a **research / educational project**. It should **not** be used for actual medical diagnosis. Always consult a qualified healthcare professional.

## 📝 License

MIT License — see [LICENSE](LICENSE)

## 👨‍💻 Author

**Chahed Nefzi**
- LinkedIn: [Chahed-nefzi](https://www.linkedin.com/in/chahed-nefzi-1a81b6218/)
- Email: shahednefzi@gmail.com

---

⭐ If this project helped you, give it a star!