"""
=====================================================================================
Modulo: dataset_fase2.py
Progetto: ML Emozioni - Fase 2 (Multimodale a 3 Segnali)

Descrizione:
Carica 3 segnali contemporaneamente (Affect, ECG, EDA).
Applica la normalizzazione Z-Score in modo INDIPENDENTE per ogni canale, per 
evitare che i Volt dell'EDA schiaccino i millivolt dell'ECG.
Restituisce un tensore di forma (3, WINDOW_SIZE).
=====================================================================================
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from config_FASE2_early import SPLIT_INDEX_PATH, WINDOW_SIZE

class PopaneDatasetMultimodal(Dataset):
    def __init__(self, split_type="train"):
        self.full_index = pd.read_csv(SPLIT_INDEX_PATH)
        self.index_df = self.full_index[self.full_index['split'] == split_type].reset_index(drop=True)
        self.window_size = WINDOW_SIZE

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        row_info = self.index_df.iloc[idx]
        file_path = row_info['file_path']
        start_row = row_info['start_row']
        label = row_info['label']

        # 1. LETTURA A 3 CANALI
        # Indici colonne nel CSV raw: 1=affect, 2=ECG, 3=EDA
        chunk = pd.read_csv(
            file_path, 
            skiprows=start_row + 1, 
            nrows=self.window_size, 
            header=None,
            usecols=[1, 2, 3] 
        )
        
        # 2. PREPARAZIONE DATI
        # Facciamo la trasposta (.T) per avere la forma (3 canali, 1000 campioni)
        raw_signals = chunk.values.T.astype(np.float32)
        raw_signals = np.nan_to_num(raw_signals) # Protezione da valori nulli (NaN)
        
        # 3. NORMALIZZAZIONE INDIPENDENTE (Il vero trucco della Fase 2)
        # Calcoliamo media e std calcolate SULL'ASSE 1 (cioè per ogni singola riga/canale)
        means = raw_signals.mean(axis=1, keepdims=True)
        stds = raw_signals.std(axis=1, keepdims=True) + 1e-8
        
        normalized_signals = (raw_signals - means) / stds

        # 4. TENSORIZZAZIONE
        # Il tensore ha già la forma corretta (3, 1000), non serve unsqueeze
        signals_tensor = torch.FloatTensor(normalized_signals)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return signals_tensor, label_tensor

# Piccolo test per verificare che non esploda nulla
if __name__ == "__main__":
    test_ds = PopaneDatasetMultimodal(split_type="train")
    segnali, etichetta = test_ds[0]
    print(f"Forma del Tensore Segnali: {segnali.shape} (Dovrebbe essere 3, 1000)")
    print(f"Etichetta: {etichetta}")