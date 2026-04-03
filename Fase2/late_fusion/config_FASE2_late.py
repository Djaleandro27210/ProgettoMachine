"""
config_FASE3_late.py
Global configuration for Phase 3 (Late Fusion).
Defines paths, signal parameters, and training hyperparameters.
"""
import os

# 1. Project Paths
BASE_DIR = "dataset"
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

INDEX_PATH = os.path.join(PROCESSED_DIR, "dataset_index.csv")
SPLIT_INDEX_PATH = os.path.join(PROCESSED_DIR, "dataset_index_split.csv")

# Late Fusion Model Checkpoints
MODEL_SAVE_PATH_LATE_ECG = os.path.join(PROCESSED_DIR, "best_late_ecg.pth")
MODEL_SAVE_PATH_LATE_EDA = os.path.join(PROCESSED_DIR, "best_late_eda.pth")
MODEL_SAVE_PATH_LATE_AFFECT = os.path.join(PROCESSED_DIR, "best_late_affect.pth")

# 2. Signal Parameters
WINDOW_SIZE = 1000  # 1 second window at 1000Hz sampling rate

# Binary classification mapping: 0 = Negative, 1 = Positive
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

# 3. Training Hyperparameters
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 80