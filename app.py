# -*- coding: utf-8 -*-
"""
Brain Tumor Detection - Professional Medical AI System
Streamlit Web Application Interface
FIXED VERSION with proper encoding and inference settings
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import load_model #ignore: E402
from PIL import Image

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Page config
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="Brain Tumor Detection System",
    page_icon="ðŸ¥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Professional Medical CSS - Dark Theme
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Montserrat:wght@400;500;600;700&display=swap');
    
    :root {
        --primary-blue: #1e293b;
        --primary-dark: #0f172a;
        --accent-cyan: #06b6d4;
        --accent-blue: #3b82f6;
        --medical-green: #10b981;
        --medical-red: #ef4444;
        --dark-navy: #0f172a;
        --dark-slate: #1e293b;
        --gray-50: #f8fafc;
        --gray-100: #f1f5f9;
        --gray-200: #e2e8f0;
        --gray-300: #cbd5e1;
        --gray-600: #475569;
        --gray-700: #334155;
        --gray-800: #1e293b;
    }
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Main content area with dark theme gradient */
    .main {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        padding-top: 1rem;
        min-height: 100vh;
    }
    
    /* Professional header with dark theme */
    .page-header {
        background: linear-gradient(135deg, var(--dark-navy) 0%, var(--dark-slate) 100%);
        padding: 3rem 2rem;
        margin: -1rem -2rem 3rem -2rem;
        text-align: center;
        border-bottom: 4px solid var(--accent-cyan);
    }
    
    .page-title {
        color: #ffffff;
        font-size: 2.75rem;
        font-weight: 800;
        margin-bottom: 0.75rem;
        letter-spacing: -1px;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .page-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.1rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }

    /* Professional Medical Sidebar */
    [data-testid="stSidebar"] {
        background: var(--dark-navy);
        border-right: none;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebar"] h3 {
        color: var(--accent-cyan);
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 2.5rem 0 1.25rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid rgba(6, 182, 212, 0.3);
    }
    
    [data-testid="stSidebar"] p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.9rem;
        line-height: 1.7;
        font-weight: 300;
    }
    
    [data-testid="stSidebar"] strong {
        color: #ffffff;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.15);
        margin: 2rem 0;
    }
    
    /* Modern metric cards */
    .metric-row {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.875rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-item {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%);
        padding: 1.25rem 1rem;
        border-left: 3px solid var(--accent-cyan);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    
    .metric-item:hover {
        transform: translateX(4px);
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(6, 182, 212, 0.15) 100%);
        border-left-color: #ffffff;
    }
    
    .metric-label {
        color: var(--accent-cyan);
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Professional result display */
    .result-display {
        max-width: 800px;
        margin: 2.5rem auto;
        text-align: center;
        padding: 3rem 2.5rem;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        border-top: 5px solid var(--accent-cyan);
        position: relative;
    }
    
    .result-display::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, var(--accent-cyan), #3b82f6);
    }
    
    .result-label {
        color: var(--accent-cyan);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .result-value {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    
    .result-tumor {
        color: var(--medical-red);
    }
    
    .result-healthy {
        color: var(--medical-green);
    }
    
    .confidence-text {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 1rem;
    }
    
    .confidence-text strong {
        color: var(--accent-cyan);
        font-weight: 700;
    }
    
    .progress-bar {
        width: 100%;
        height: 8px;
        background: rgba(15, 23, 42, 0.5);
        margin-top: 2rem;
        overflow: hidden;
        position: relative;
    }
    
    .progress-fill {
        height: 100%;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .progress-tumor {
        background: linear-gradient(90deg, var(--medical-red), #dc2626);
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
    }
    
    .progress-healthy {
        background: linear-gradient(90deg, var(--medical-green), #059669);
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }
    
    /* Image sections with modern styling */
    .image-container {
        margin: 4rem 0;
        padding: 2rem;
        background: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(6, 182, 212, 0.2);
    }
    
    .image-label {
        color: var(--accent-cyan);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-bottom: 1.25rem;
        text-align: center;
        padding-bottom: 0.75rem;
        border-bottom: 3px solid var(--accent-cyan);
    }
    
    /* Professional info boxes */
    .info-note {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        border-left: 5px solid var(--accent-cyan);
        padding: 2rem 1.75rem;
        margin: 2.5rem 0;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.8;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.15);
        backdrop-filter: blur(10px);
    }
    
    .info-note strong {
        color: var(--accent-cyan);
        display: block;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        font-weight: 700;
    }
    
    .warning-note {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        border-left: 5px solid var(--medical-red);
        padding: 2rem 1.75rem;
        margin: 2.5rem 0;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.8;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
        backdrop-filter: blur(10px);
    }
    
    .warning-note strong {
        color: var(--medical-red);
        display: block;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        font-weight: 700;
    }
    
    .success-note {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        border-left: 5px solid var(--medical-green);
        padding: 2rem 1.75rem;
        margin: 2.5rem 0;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.8;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
        backdrop-filter: blur(10px);
    }
    
    .success-note strong {
        color: var(--medical-green);
        display: block;
        margin-bottom: 1rem;
        font-size: 1.1rem;
        font-weight: 700;
    }
    
    /* Styling for markdown inside notes */
    .info-note + div p,
    .warning-note + div p,
    .success-note + div p {
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.8;
        margin-bottom: 1rem;
    }
    
    .info-note + div strong,
    .warning-note + div strong,
    .success-note + div strong {
        color: var(--accent-cyan);
        font-weight: 600;
    }
    
    .warning-note + div strong {
        color: #fca5a5;
    }
    
    .success-note + div strong {
        color: #6ee7b7;
    }
    
    /* Expander content styling */
    .streamlit-expanderContent p {
        color: #e2e8f0;
        line-height: 1.8;
        margin-bottom: 1rem;
    }
    
    .streamlit-expanderContent strong {
        color: var(--accent-cyan);
        font-weight: 600;
    }
    
    /* Error message styling */
    .stAlert {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(30, 41, 59, 0.8) 100%);
        border-left: 4px solid var(--medical-red);
        color: #fca5a5;
        backdrop-filter: blur(10px);
    }
    
    .stAlert [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0;
        line-height: 1.7;
    }
    
    .stAlert [data-testid="stMarkdownContainer"] strong {
        color: var(--medical-red);
        font-weight: 600;
    }
    
    .stAlert [data-testid="stMarkdownContainer"] ul {
        color: #cbd5e1;
    }
    
    /* Modern file uploader */
    [data-testid="stFileUploader"] {
        border: 3px dashed var(--accent-cyan);
        padding: 3rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(51, 65, 85, 0.4) 100%);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        box-shadow: 0 8px 24px rgba(6, 182, 212, 0.2);
    }
    
    [data-testid="stFileUploader"] label {
        color: var(--accent-cyan) !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: #94a3b8 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        border: 2px solid var(--accent-cyan);
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--accent-cyan);
        transition: all 0.2s ease;
        backdrop-filter: blur(10px);
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.7) 100%);
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.2);
    }
    
    .streamlit-expanderContent {
        background: rgba(30, 41, 59, 0.4);
        border-left: 2px solid var(--accent-cyan);
        border-right: 2px solid var(--accent-cyan);
        border-bottom: 2px solid var(--accent-cyan);
    }
    
    /* Professional buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--dark-navy) 0%, var(--dark-slate) 100%);
        color: white;
        border: none;
        padding: 0.875rem 2.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--dark-slate) 0%, var(--accent-cyan) 100%);
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.4);
        transform: translateY(-2px);
    }
    
    /* Column styling */
    [data-testid="column"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(51, 65, 85, 0.4) 100%);
        padding: 2rem 1.5rem;
        border: 2px solid rgba(6, 182, 212, 0.3);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="column"]:hover {
        border-color: var(--accent-cyan);
        box-shadow: 0 4px 16px rgba(6, 182, 212, 0.2);
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
    }
    
    /* Sidebar warning note */
    [data-testid="stSidebar"] .warning-note {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid var(--medical-red);
        padding: 1.25rem;
        margin: 2rem 0;
        color: #ffffff;
        font-size: 0.85rem;
        line-height: 1.7;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] .warning-note strong {
        color: var(--medical-red);
        display: block;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
        font-weight: 700;
    }
    
    /* Background consistency */
    html, body, [data-testid="stApp"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    }
    
    section.main > div {
        background: transparent !important;
    }
    
    /* Remove Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--accent-cyan) !important;
    }
    
    /* Professional spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    </style>
""", unsafe_allow_html=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Professional Header
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
    <div class="page-header">
        <div class="page-title">Brain Tumor Detection System</div>
        <div class="page-subtitle">AI-Powered Medical Imaging Analysis with Explainable Grad-CAM Technology</div>
    </div>
""", unsafe_allow_html=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Model Loading
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
IMG_SIZE = 150
GRADCAM_LAYER = 'conv4'
MODEL_CONFIG_PATH = "model_config.json"
DEFAULT_DECISION_THRESHOLD = 0.52
LOW_CONFIDENCE_MARGIN = 0.03
UNCERTAIN_MARGIN = 0.04


@st.cache_data
def load_decision_threshold():
    if not os.path.exists(MODEL_CONFIG_PATH):
        return DEFAULT_DECISION_THRESHOLD
    try:
        with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        value = float(cfg.get("decision_threshold", DEFAULT_DECISION_THRESHOLD))
        # Keep threshold in a safe range for binary healthy-vs-tumor decision
        return min(max(value, 0.50), 0.95)
    except Exception:
        return DEFAULT_DECISION_THRESHOLD


DECISION_THRESHOLD = load_decision_threshold()

@st.cache_resource
def load_trained_model():
    try:
        model = load_model('best_brain_tumor_model.h5')
        model.trainable = False  # FIXED: Set to False for inference to prevent BatchNorm issues
        return model
    except Exception as e:
        st.error(f"âš ï¸ Model Loading Error: {e}")
        st.info("""
            **Model not found!** Please train the model first:
            
            1. Run: `python detection.py`
            2. Wait for training to complete
            3. Then run: `streamlit run app.py`
        """)
        return None

model = load_trained_model()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Grad-CAM Functions
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def generate_gradcam(model, img_array, layer_name=GRADCAM_LAYER):
    """Generate Grad-CAM heatmap"""
    _ = model(img_array)
    
    conv_layer_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(layer_name).output
    )
    
    classifier_input = tf.keras.Input(shape=model.get_layer(layer_name).output.shape[1:])
    x = classifier_input
    
    found_target = False
    for layer in model.layers:
        if found_target:
            x = layer(x)
        if layer.name == layer_name:
            found_target = True
    
    classifier_model = tf.keras.Model(inputs=classifier_input, outputs=x)
    
    with tf.GradientTape() as tape:
        conv_outputs = conv_layer_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        top_class_channel = predictions[:, 0]
    
    grads = tape.gradient(top_class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    
    return heatmap


def overlay_heatmap(heatmap, original_img, alpha=0.4):
    """Overlay heatmap on original image"""
    if original_img.max() <= 1.0:
        original_img = (original_img * 255).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)


def preprocess_mri_image(img_rgb):
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    img_preprocessed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    img_normalized = (img_preprocessed / 255.0).astype(np.float32)
    return img_resized, img_preprocessed, img_normalized


def assess_mri_likelihood(img_resized):
    r_channel = img_resized[:, :, 0]
    g_channel = img_resized[:, :, 1]
    b_channel = img_resized[:, :, 2]

    rg_diff = np.abs(r_channel.astype(float) - g_channel.astype(float)).mean()
    rb_diff = np.abs(r_channel.astype(float) - b_channel.astype(float)).mean()
    gb_diff = np.abs(g_channel.astype(float) - b_channel.astype(float)).mean()
    avg_channel_diff = (rg_diff + rb_diff + gb_diff) / 3.0

    grayscale = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    gray_similarity = np.corrcoef(grayscale.flatten(), r_channel.flatten())[0, 1]
    if np.isnan(gray_similarity):
        gray_similarity = 0.0

    hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].mean()
    mean_intensity = float(np.mean(img_resized))
    std_intensity = float(np.std(img_resized))

    edges = cv2.Canny(grayscale, 50, 150)
    edge_density = float(np.sum(edges > 0) / (IMG_SIZE * IMG_SIZE))

    checks = {
        "grayscale_like": avg_channel_diff < 45,
        "low_saturation": saturation < 70,
        "gray_similarity": gray_similarity > 0.70,
        "intensity_range": 10 < mean_intensity < 245 and std_intensity > 8,
        "edge_density": 0.003 < edge_density < 0.60,
    }
    passed = sum(int(v) for v in checks.values())
    is_likely_mri = passed >= 3

    reasons = []
    if not checks["grayscale_like"]:
        reasons.append("Color channels are very different from each other.")
    if not checks["low_saturation"]:
        reasons.append("Color saturation is high for a typical MRI.")
    if not checks["gray_similarity"]:
        reasons.append("Image is weakly correlated with grayscale structure.")
    if not checks["intensity_range"]:
        reasons.append("Intensity distribution is unusual for MRI-like scans.")
    if not checks["edge_density"]:
        reasons.append("Edge density is outside common MRI-like range.")

    metrics = {
        "avg_channel_diff": avg_channel_diff,
        "gray_similarity": gray_similarity,
        "saturation": saturation,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "edge_density": edge_density,
        "checks_passed": passed,
    }
    return is_likely_mri, reasons, metrics

# Professional Sidebar

with st.sidebar:
    st.markdown("### PERFORMANCE METRICS")
    
    st.markdown("""
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">97.78%</div>  
            </div>
            <div class="metric-item">
                <div class="metric-label">Test Loss</div>
                <div class="metric-value">0.072</div>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-label">Sensitivity</div>
                <div class="metric-value">96.76%</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Specificity</div>
                <div class="metric-value">100%</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### SYSTEM DETAILS")
    st.markdown("""
        **Model:** Custom CNN  
        **Parameters:** 3.6M  
        **Input:** 150Ã—150 RGB  
        **Framework:** TensorFlow  
        **Visualization:** Grad-CAM
    """)
    
    st.markdown("### DATASET INFO")
    st.markdown("""
        **Source:** Kaggle Brain MRI  
        **Total Images:** 5,712  
        **Split:** 70/15/15  
        **Classes:** Binary (Tumor/No Tumor)
    """)
    
    st.markdown("---")
    
    st.markdown("""
        <div class="warning-note">
            <strong>MEDICAL DISCLAIMER</strong>
            This is a research and educational tool only. Not approved for clinical diagnosis. Always consult qualified medical professionals for health decisions.
        </div>
    """, unsafe_allow_html=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Main Content
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if model is None:
    st.stop()

uploaded_file = st.file_uploader(
    "Upload Brain MRI Scan",
    type=['jpg', 'png', 'jpeg'],
    help="Supported: JPG, PNG, JPEG â€¢ Recommended: High-resolution axial MRI scans"
)

if uploaded_file is not None:
    # Process image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    
    # MRI plausibility check (tolerant to valid dark/low-contrast scans)
    is_likely_mri, mri_reasons, mri_metrics = assess_mri_likelihood(img_resized)

    if not is_likely_mri:
        st.markdown("""
            <div class="warning-note">
                <strong>INVALID IMAGE DETECTED</strong>
            </div>
        """, unsafe_allow_html=True)

        rejection_reasons = []
        if mri_metrics["avg_channel_diff"] >= 45:
            rejection_reasons.append(f"• Image appears strongly colored (channel difference: {mri_metrics['avg_channel_diff']:.1f}, expected < 45)")
        if mri_metrics["saturation"] >= 70:
            rejection_reasons.append(f"• High color saturation detected ({mri_metrics['saturation']:.1f}, expected < 70)")
        if mri_metrics["gray_similarity"] <= 0.70:
            rejection_reasons.append(f"• Weak grayscale similarity ({mri_metrics['gray_similarity']:.2f}, expected > 0.70)")
        if not (10 < mri_metrics["mean_intensity"] < 245 and mri_metrics["std_intensity"] > 8):
            rejection_reasons.append(f"• Intensity outside MRI-like range (mean: {mri_metrics['mean_intensity']:.1f}, std: {mri_metrics['std_intensity']:.1f})")
        if not (0.003 < mri_metrics["edge_density"] < 0.60):
            rejection_reasons.append(f"• Edge pattern inconsistent with MRI-like scans (density: {mri_metrics['edge_density']:.3f})")

        if not rejection_reasons and mri_reasons:
            rejection_reasons = [f"• {reason}" for reason in mri_reasons]

        st.error(f"""
            **Warning:** The uploaded image does not appear to be a valid brain MRI scan.

            **Issues Detected:**
            {chr(10).join(rejection_reasons)}

            **Common Problems:**
            - Image is a color photograph or illustration
            - Image is a screenshot or document
            - Image brightness/contrast inconsistent with medical imaging
            - Image lacks the characteristic grayscale appearance of MRI scans

            **Please upload:**
            - Genuine brain MRI scans (T1, T2, FLAIR sequences)
            - Grayscale medical imaging files
            - DICOM-converted images in JPG/PNG format
            - Images should show brain cross-sections in grayscale

            **Requirements:**
            - Format: JPG, PNG, JPEG
            - Content: Axial, sagittal, or coronal brain MRI scan
            - Quality: High-resolution medical imaging (not photos of screens)
            - Appearance: Grayscale with visible brain structures
        """)
        st.stop()
    img_batch = np.expand_dims(img_normalized, axis=0)

    # Prediction
    with st.spinner('Analyzing MRI scan with AI...'):
        prediction = model.predict(img_batch, verbose=0)[0][0]

        if prediction >= (DECISION_THRESHOLD + UNCERTAIN_MARGIN):
            result = "No Tumor Detected"
            confidence = prediction * 100
            result_class = "result-healthy"
            progress_class = "progress-healthy"
        elif prediction <= (DECISION_THRESHOLD - UNCERTAIN_MARGIN):
            result = "Tumor Detected"
            confidence = (1 - prediction) * 100
            result_class = "result-tumor"
            progress_class = "progress-tumor"
        else:
            result = "Inconclusive - Review Needed"
            confidence = 100 - (abs(float(prediction) - DECISION_THRESHOLD) * 100)
            result_class = "result-healthy"
            progress_class = "progress-healthy"
        is_borderline = abs(float(prediction) - DECISION_THRESHOLD) <= LOW_CONFIDENCE_MARGIN

        # Grad-CAM
        heatmap = generate_gradcam(model, img_batch)
        overlay = overlay_heatmap(heatmap, img_resized)

    # Professional Result Display
    st.markdown(f"""
        <div class="result-display">
            <div class="result-label">DIAGNOSTIC RESULT</div>
            <div class="result-value {result_class}">{result}</div>
            <div class="confidence-text">Confidence Level: <strong>{confidence:.2f}%</strong></div>
            <div class="progress-bar">
                <div class="progress-fill {progress_class}" style="width: {confidence}%;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if is_borderline:
        st.warning(
            f"Borderline prediction: model score={prediction:.4f}, threshold={DECISION_THRESHOLD:.2f}. "
            "This case is close to the decision boundary and should be reviewed carefully."
        )
    if result == "Inconclusive - Review Needed":
        st.info(
            "Recommendation: do not ignore this image. Re-check with another MRI slice/sequence and confirm with medical review."
        )

    # Image Grid - Professional Layout
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="image-label">Original MRI Scan</div>', unsafe_allow_html=True)
        st.image(img_resized, use_container_width=True)

    with col2:
        st.markdown('<div class="image-label">Neural Activation Map</div>', unsafe_allow_html=True)
        heatmap_vis = cv2.applyColorMap(
            (cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE)) * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap_vis = cv2.cvtColor(heatmap_vis, cv2.COLOR_BGR2RGB)
        st.image(heatmap_vis, use_container_width=True)

    with col3:
        st.markdown('<div class="image-label">Grad-CAM Overlay</div>', unsafe_allow_html=True)
        st.image(overlay, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Interpretation Guide
    with st.expander("Interpretation Guide"):
        st.markdown("""
            <div class="info-note">
                <strong>How to Read the Heatmap Visualization:</strong>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            **Red/Warm Regions:** Areas that strongly influenced the AI model's diagnostic decision
            
            **Blue/Cool Regions:** Areas with minimal impact on the classification outcome
            
            **Clinical Correlation:** For tumor predictions, red activation zones should correspond with visible anatomical anomalies
            
            **High Confidence Threshold:** Results exceeding 90% confidence warrant immediate clinical review and validation
        """)

    # Technical Details
    with st.expander("Technical Specifications"):
        st.markdown(f"""
        **Raw Prediction Score:** `{prediction:.6f}`  
        **Classification Threshold:** `{DECISION_THRESHOLD:.2f}`  
        **Grad-CAM Target Layer:** `{GRADCAM_LAYER}`  
        **Input Dimensions:** `{IMG_SIZE}Ã—{IMG_SIZE}Ã—3`  
        **Model Architecture:** Custom Convolutional Neural Network  
        **Total Parameters:** 6.8 Million  
        **Framework:** TensorFlow 2.x + Keras API
        """)

    # Clinical Recommendations
    if result == "Tumor Detected":
        st.markdown("""
            <div class="warning-note">
                <strong>RECOMMENDED CLINICAL ACTIONS:</strong>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            **1. Immediate Specialist Review:** Consult a board-certified neuroradiologist or neurologist
            
            **2. Additional Imaging Studies:** Consider contrast-enhanced MRI, CT scan, or PET scan as clinically indicated
            
            **3. Clinical Correlation:** Correlate findings with patient symptoms, neurological examination, and medical history
            
            **4. Critical Warning:** DO NOT base treatment decisions solely on AI predictions
            
            **5. Follow-up Protocol:** Schedule comprehensive diagnostic workup and multidisciplinary team review
        """)
    else:
        st.markdown("""
            <div class="success-note">
                <strong>CLINICAL INTERPRETATION:</strong>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            The AI system has not detected tumor indicators in this MRI scan. However, this is a screening and educational tool that requires professional medical validation.
            
            **Important Next Steps:** Regular monitoring, scheduled follow-up imaging, and ongoing clinical surveillance remain essential components of comprehensive patient care and neurological health management.
        """)

else:
    # Professional Landing State
    st.markdown("""
        <div class="info-note">
            <strong>GETTING STARTED</strong><br><br>
            Upload a brain MRI scan using the file uploader above to begin AI-assisted diagnostic analysis with explainable Grad-CAM visualization technology.<br><br>
            <strong>Supported Formats:</strong> JPG, PNG, JPEG<br><br>
            <strong>Recommended:</strong> High-resolution axial T1 or T2-weighted MRI scans<br><br>
            <strong>File Size Limit:</strong> 200MB per file
        </div>
    """, unsafe_allow_html=True)

