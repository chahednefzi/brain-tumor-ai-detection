import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras_preprocessing.image import ImageDataGenerator # type: ignore
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import cv2
from PIL import Image

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
ROOT_DIR = "dataset/Training"
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 25

print("=" * 60)
print("BRAIN TUMOR DETECTION - BINARY CLASSIFICATION")
print("Dataset: Brain Tumor MRI (Kaggle)")
print("=" * 60)

# ─────────────────────────────────────────────
# Count images per class
# ─────────────────────────────────────────────
number_of_images = {}
for dir in os.listdir(ROOT_DIR):
    number_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR, dir)))

print("\nOriginal classes distribution:")
for class_name, count in number_of_images.items():
    print(f"  {class_name}: {count} images")

# ─────────────────────────────────────────────
# Binary labeling
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONVERTING TO BINARY CLASSIFICATION")
print("=" * 60)
print("  Tumor   (0): glioma + meningioma + pituitary")
print("  Healthy (1): notumor")

image_paths = []
labels = []

for class_name in os.listdir(ROOT_DIR):
    class_path = os.path.join(ROOT_DIR, class_name)
    for img_name in os.listdir(class_path):
        image_paths.append(os.path.join(class_path, img_name))
        labels.append(1 if class_name == 'notumor' else 0)

print(f"\nTotal images collected: {len(image_paths)}")

tumor_count = labels.count(0)
healthy_count = labels.count(1)
binary_distribution = {'Tumor': tumor_count, 'Healthy': healthy_count}

print("\nBinary distribution:")
for k, v in binary_distribution.items():
    print(f"  {k}: {v}")

# ─────────────────────────────────────────────
# Plots: distribution & samples
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.bar(binary_distribution.keys(), binary_distribution.values(), color=['red', 'green'])
plt.title('Binary Distribution: Tumor vs Healthy')
plt.xlabel('Classes')
plt.ylabel('Number of Images')
for i, (key, value) in enumerate(binary_distribution.items()):
    plt.text(i, value + 50, str(value), ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('binary_distribution.png')
print("\n✓ binary_distribution.png saved")

def display_sample_images():
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    fig.suptitle('Sample Images: Tumor Types vs Healthy', fontsize=16)
    classes = os.listdir(ROOT_DIR)
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(ROOT_DIR, class_name)
        images = os.listdir(class_path)[:3]
        for i in range(3):
            img = cv2.imread(os.path.join(class_path, images[i]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i, idx].imshow(img)
            axes[i, idx].set_title(class_name, fontsize=10)
            axes[i, idx].axis('off')
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("✓ sample_images.png saved")

display_sample_images()

# ─────────────────────────────────────────────
# Train / Val / Test split
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA PREPARATION")
print("=" * 60)

X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels, test_size=0.3, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\n  Training   : {len(X_train)}")
print(f"  Validation : {len(X_val)}")
print(f"  Test       : {len(X_test)}")

# ─────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img / 255.0

print("\nLoading images...")
X_train_images = np.array([load_and_preprocess_image(p) for p in X_train])
y_train = np.array(y_train)
X_val_images   = np.array([load_and_preprocess_image(p) for p in X_val])
y_val   = np.array(y_val)
X_test_images  = np.array([load_and_preprocess_image(p) for p in X_test])
y_test  = np.array(y_test)

print(f"  Train shape : {X_train_images.shape}")
print(f"  Val shape   : {X_val_images.shape}")
print(f"  Test shape  : {X_test_images.shape}")

# ─────────────────────────────────────────────
# Data augmentation
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DATA AUGMENTATION")
print("=" * 60)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train_images)
print("✓ Data augmentation configured")

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL ARCHITECTURE - CNN")
print("=" * 60)

model = models.Sequential([
    layers.Conv2D(32,  (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3), name='conv1'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,  (3,3), activation='relu', name='conv2'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu', name='conv3'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(256, (3,3), activation='relu', name='conv4'),  # ← Grad-CAM target
    layers.BatchNormalization(name='last_bn'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nModel Summary:")
model.summary()

# ─────────────────────────────────────────────
# Callbacks & Training
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ModelCheckpoint('best_brain_tumor_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
]

history = model.fit(
    datagen.flow(X_train_images, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_val_images, y_val),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

model.save('brain_tumor_model_final.h5')
print("\n✓ brain_tumor_model_final.h5 saved")

# ─────────────────────────────────────────────
# Training history plot
# ─────────────────────────────────────────────
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'],     label='Training',   marker='o', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', marker='s', linewidth=2)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'],     label='Training',   marker='o', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', marker='s', linewidth=2)
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    print("✓ training_history.png saved")

plot_history(history)

# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL EVALUATION ON TEST SET")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(X_test_images, y_test, verbose=0)
print(f"\n  Test Accuracy : {test_accuracy * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")

predictions = model.predict(X_test_images, verbose=0)
predicted_classes = (predictions > 0.5).astype(int).flatten()

cm = confusion_matrix(y_test, predicted_classes)
class_labels = ['Tumor', 'Healthy']

print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes, target_names=class_labels))

# Sensitivity / Specificity (label 0 = Tumor, 1 = Healthy)
tumor_tp  = cm[0][0]   # Actual Tumor  → Predicted Tumor
tumor_fn  = cm[0][1]   # Actual Tumor  → Predicted Healthy (missed!)
healthy_tp = cm[1][1]  # Actual Healthy → Predicted Healthy
healthy_fp = cm[1][0]  # Actual Healthy → Predicted Tumor

sensitivity = tumor_tp / (tumor_tp + tumor_fn)
specificity = healthy_tp / (healthy_tp + healthy_fp)

print(f"  Sensitivity (Tumor Detection Rate)      : {sensitivity * 100:.2f}%")
print(f"  Specificity (Healthy Identification Rate): {specificity * 100:.2f}%")

# Confusion matrix plot
def plot_confusion_matrix(cm, class_labels):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Binary Classification', fontsize=16, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, fontsize=12)
    plt.yticks(tick_marks, class_labels, fontsize=12)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✓ confusion_matrix.png saved")

plot_confusion_matrix(cm, class_labels)

# ═══════════════════════════════════════════════════════════
# GRAD-CAM
# ═══════════════════════════════════════════════════════════
def generate_gradcam(model, img_array, layer_name='conv4'):
    """
    Generate a Grad-CAM heatmap for a single preprocessed image.
    """
    # Ensure model is built
    _ = model(img_array)
    
    last_conv_layer = model.get_layer(layer_name)
    
    # CRITICAL: Use model.outputs[0] instead of model.output for Keras 3
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    
    if grads is None:
        raise ValueError(f"Gradients are None. Check layer '{layer_name}'.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


def overlay_heatmap(heatmap, original_img, alpha=0.4):
    """
    Resize the heatmap to match the original image and blend it on top.

    Args:
        heatmap     : 2D array [0,1] from generate_gradcam
        original_img: RGB image as uint8 (0-255) or float (0-1)
        alpha       : blend strength (0 = original only, 1 = heatmap only)

    Returns:
        superimposed: RGB uint8 image with the heatmap overlay
    """
    # Ensure uint8
    if original_img.max() <= 1.0:
        original_img = (original_img * 255).astype(np.uint8)

    # Resize heatmap → image spatial size
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

    # Convert [0,1] heatmap to a colored map (JET: blue=low → red=high)
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    # applyColorMap returns BGR; convert to RGB
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Alpha blend
    superimposed = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed


def predict_image(model, img_path):
    """
    Predict whether an MRI contains a tumor, and generate a Grad-CAM heatmap.

    Args:
        model   : trained Keras model
        img_path: path to the image file

    Returns:
        result    : 'Tumor' or 'Healthy'
        confidence: percentage (float)
    """
    # Load & preprocess
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    # Prediction
    prediction = model.predict(img_batch, verbose=0)[0][0]
    if prediction > 0.5:
        result     = "Healthy"
        confidence = prediction * 100
    else:
        result     = "Tumor"
        confidence = (1 - prediction) * 100

    # Grad-CAM
    heatmap = generate_gradcam(model, img_batch)
    overlay = overlay_heatmap(heatmap, img_resized)

    # ── 3-panel figure ──
    color = 'green' if result == "Healthy" else 'red'
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Prediction: {result}  |  Confidence: {confidence:.2f}%',
                 fontsize=16, fontweight='bold', color=color)

    axes[0].imshow(img_resized)
    axes[0].set_title('Original MRI', fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=13)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Heatmap Overlay', fontsize=13)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*40}")
    print(f"  Prediction : {result}")
    print(f"  Confidence : {confidence:.2f}%")
    print(f"{'='*40}")

    return result, confidence


# ─────────────────────────────────────────────
# Sample predictions + Grad-CAM grid
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

# 8-image prediction grid (no heatmap, quick overview)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Sample Test Predictions', fontsize=16, fontweight='bold')

sample_indices = np.random.choice(len(X_test), 8, replace=False)
for idx, ax in enumerate(axes.flat):
    i = sample_indices[idx]
    true_label = 'Healthy' if y_test[i] == 1 else 'Tumor'
    pred_label = 'Healthy' if predicted_classes[i] == 1 else 'Tumor'
    color = 'green' if true_label == pred_label else 'red'
    ax.imshow(X_test_images[i])
    ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=300)
print("✓ sample_predictions.png saved")

# Grad-CAM grid: 4 test images side-by-side with their heatmaps
print("\nGenerating Grad-CAM heatmap grid...")
gradcam_indices = np.random.choice(len(X_test), 4, replace=False)

fig, axes = plt.subplots(4, 3, figsize=(12, 16))
fig.suptitle('Grad-CAM Explanations — What the model sees', fontsize=17, fontweight='bold', y=0.98)

for row, i in enumerate(gradcam_indices):
    img_input = np.expand_dims(X_test_images[i], axis=0)

    true_label = 'Healthy' if y_test[i] == 1 else 'Tumor'
    pred_label = 'Healthy' if predicted_classes[i] == 1 else 'Tumor'
    correct    = true_label == pred_label
    color      = 'green' if correct else 'red'
    status     = '✓' if correct else '✗'

    # Generate heatmap & overlay
    heatmap = generate_gradcam(model, img_input)
    overlay = overlay_heatmap(heatmap, X_test_images[i])

    # Col 0 — Original
    axes[row, 0].imshow(X_test_images[i])
    axes[row, 0].set_title(f'{status} True: {true_label} | Pred: {pred_label}',
                           color=color, fontsize=11, fontweight='bold')
    axes[row, 0].axis('off')

    # Col 1 — Raw heatmap
    axes[row, 1].imshow(heatmap, cmap='jet')
    axes[row, 1].set_title('Heatmap', fontsize=11)
    axes[row, 1].axis('off')

    # Col 2 — Overlay
    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title('Overlay', fontsize=11)
    axes[row, 2].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('gradcam_explanations.png', dpi=300)
print("✓ gradcam_explanations.png saved")

# ─────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("PROJECT COMPLETE!")
print("=" * 60)
print("\nGenerated files:")
print("  ✓ binary_distribution.png")
print("  ✓ sample_images.png")
print("  ✓ training_history.png")
print("  ✓ confusion_matrix.png")
print("  ✓ sample_predictions.png")
print("  ✓ gradcam_explanations.png")
print("  ✓ best_brain_tumor_model.h5")
print("  ✓ brain_tumor_model_final.h5")
print("\nTo predict a new image:")
print("  from keras.models import load_model")
print("  from detection import predict_image")
print("  model = load_model('best_brain_tumor_model.h5')")
print("  predict_image(model, 'path/to/your/image.jpg')")
print("=" * 60)