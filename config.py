"""
config.py
Global configuration and hyperparameters for the project.
Defines core paths, dataset mappings, and training constants.
"""
import os

# 1. Project Paths
BASE_DIR = "dataset"
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

INDEX_PATH = os.path.join(PROCESSED_DIR, "dataset_index.csv")
SPLIT_INDEX_PATH = os.path.join(PROCESSED_DIR, "dataset_index_split.csv")
MODEL_SAVE_PATH = os.path.join(PROCESSED_DIR, "best_emotion_model.pth")

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
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 80