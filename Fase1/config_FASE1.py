
"""
=========================================================
File: config.py
Description: Global constants and project parameters.
=========================================================
"""
import os

# ==========================================
# 1. PATHS
# ==========================================
BASE_DIR = "dataset"
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

INDEX_PATH = os.path.join(PROCESSED_DIR, "dataset_index.csv")
SPLIT_INDEX_PATH = os.path.join(PROCESSED_DIR, "dataset_index_split.csv")

# ==========================================
# 2. SIGNAL AND DATASET PARAMETERS
# ==========================================
# How many samples form a "slice" (1000 = 1 second at 1000Hz)
WINDOW_SIZE = 1000 

# The emotions we are interested in and their mapping (0 = Negative, 1 = Positive)
EMOTION_MAP = {
    'amusement': 1,
    'positive_emotion_low_approach': 1,
    'positive_emotion_high_approach': 1,
    'excitement': 1,
    'gratitude': 1,
    'tenderness': 1,
    'anger': 0,
    'threat': 0,
    'disgust':0,
    'sadness': 0,
    'fear': 0
}
# ==========================================
# 3. TRAINING PARAMETERS (Hyperparameters)
# ==========================================
# WEIGHTS FOR HANDLING IMBALANCED CLASSES (pos_weight for BCEWithLogitsLoss)
WEIGHT_DEECAY = 1e-4
BATCH_SIZE = 32

# The "learning rate": how quickly the network changes its mind (too high = oscillates, too low = never learns)
LEARNING_RATE = 0.0001

# Quante volte la rete vedrà l'INTERO dataset 
EPOCHS = 80

MODEL_SAVE_PATH = os.path.join("modelli", "best_model_FASE1.pth")
