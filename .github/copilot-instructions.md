# Brain Tumor AI Detection - Copilot Instructions

## Project Overview
This is a TensorFlow/Keras-based binary classification system for detecting brain tumors from MRI scans. It converts a multi-class dataset (glioma, meningioma, pituitary, notumor) into binary classification (Tumor vs Healthy).

## Architecture
- **Model**: CNN with 4 convolutional blocks (32→64→128→256 filters), BatchNormalization, MaxPooling, followed by Flatten, Dropout(0.5), Dense(512), Dropout(0.3), Dense(1, sigmoid)
- **Input**: 150x150 RGB images, normalized to [0,1]
- **Output**: Binary prediction (0=Tumor, 1=Healthy) with confidence score

## Data Handling
- **Dataset Structure**: `dataset/Training/` and `dataset/Testing/` with subfolders: glioma/, meningioma/, notumor/, pituitary/
- **Binary Mapping**: glioma + meningioma + pituitary → Tumor (0), notumor → Healthy (1)
- **Preprocessing**: Resize to (150,150), normalize by 255.0
- **Augmentation**: Rotation(20°), shift(0.2), shear(0.2), zoom(0.2), horizontal flip

## Training Workflow
- **Split**: 70% train, 15% val, 15% test with stratification
- **Hyperparameters**: BATCH_SIZE=32, EPOCHS=25
- **Optimizer**: Adam, Loss: binary_crossentropy, Metric: accuracy
- **Callbacks**: EarlyStopping(patience=7), ModelCheckpoint('best_brain_tumor_model.h5'), ReduceLROnPlateau(factor=0.2, patience=3)
- **Output Files**: Saves 'brain_tumor_model_final.h5' and various PNG plots (distribution, history, confusion matrix, samples)

## Inference
- **Function**: `predict_image(img_path)` returns (result, confidence) and saves 'prediction_result.png'
- **Threshold**: >0.5 → Healthy, else Tumor
- **Visualization**: Displays image with prediction overlay

## Key Files
- `detection.py`: Main training and evaluation script
- `requirements.txt`: Dependencies (tensorflow, numpy, matplotlib, scikit-learn, opencv-python, pillow)
- `dataset/`: Training/Testing images in class subfolders
- `model/`: Directory for saved models (create if needed)

## Development Notes
- Always use absolute paths for image loading
- Models saved in HDF5 format (.h5)
- Evaluation includes confusion matrix, sensitivity, specificity
- Sample predictions generate 8 random test images with true/pred labels

## Error Handling
The project has minimal error handling. It assumes the dataset directory exists and all images load successfully. Common failure points:
- Missing dataset: Script fails with FileNotFoundError when listing directories
- Corrupted images: cv2.imread returns None, causing errors in resize/processing
- Model loading: If .h5 files are missing, predict_image will fail
For robustness, wrap file operations in try-except blocks and validate image loading before processing.</content>
<parameter name="filePath">c:\brain-tumor-ai-detection\.github\copilot-instructions.md