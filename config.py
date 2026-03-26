"FILE PER LE VARIABILI GLOBALI DI CONFIGURAZIONE"


"""
=========================================================
File: config.py
Descrizione: Costanti globali e parametri del progetto.
Modifica i valori qui per applicarli a tutto il codice!
=========================================================
"""
import os

# ==========================================
# 1. PERCORSI (PATHS)
# ==========================================
BASE_DIR = "dataset"
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

INDEX_PATH = os.path.join(PROCESSED_DIR, "dataset_index.csv")
SPLIT_INDEX_PATH = os.path.join(PROCESSED_DIR, "dataset_index_split.csv")

# ==========================================
# 2. PARAMETRI DEL SEGNALE E DATASET
# ==========================================
# Quanti campioni formano una "fettina" (1000 = 1 secondo a 1000Hz)
# Se vuoi provare i 130 del paper originale, ti basta cambiare questo numero!
WINDOW_SIZE = 1000 

# Le emozioni che ci interessano e il loro mapping (0 = Negative, 1 = Positive)
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
# 3. PARAMETRI DI TRAINING (Hyperparameters)
# ==========================================
#PESI PER GESTIONE CLASSI SBILANCIATE (pos_weight per BCEWithLogitsLoss)
WEIGHT_DEECAY = 1e-4
# Quante fettine legge la rete prima di aggiornare i pesi
BATCH_SIZE = 32

# Il "tasso di apprendimento": quanto velocemente la rete cambia idea (troppo alto = sbanda, troppo basso = non impara mai)
LEARNING_RATE = 0.0001

# Quante volte la rete vedrà l'INTERO dataset (tutte le 86.850 fettine)
EPOCHS = 80

# Dove salveremo il "cervello" della rete una volta addestrata
MODEL_SAVE_PATH = os.path.join(PROCESSED_DIR, "best_emotion_model.pth")
