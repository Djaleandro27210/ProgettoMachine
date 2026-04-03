"""
config_FASE1.py
Global configuration and hyperparameters for Phase 1 (ECG-only model).
"""
import os

# Paths
BASE_DIR = "dataset"
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
MODELS_DIR = "modelli"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

INDEX_PATH = os.path.join(PROCESSED_DIR, "dataset_index.csv")
SPLIT_INDEX_PATH = os.path.join(PROCESSED_DIR, "dataset_index_split.csv")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_model_FASE1.pth")

# Dataset mapping
WINDOW_SIZE = 1000  # 1 second window at 1000Hz

# 0: Negative, 1: Positive
EMOTION_MAP = {
    'amusement': 1,
    'positive_emotion_low_approach': 1,
    'positive_emotion_high_approach': 1,
    'excitement': 1,
    'gratitude': 1,
    'tenderness': 1,
    'anger': 0,
    'threat': 0,
    'disgust': 0,
    'sadness': 0,
    'fear': 0
}

# Training params
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
EPOCHS = 80

# Callbacks thresholds
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 4
LR_SCHEDULER_FACTOR = 0.5
CLASSIFICATION_THRESHOLD = 0.5