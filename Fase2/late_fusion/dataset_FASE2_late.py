"""
=====================================================================================
Modulo: dataset_FASE2_late.py
Progetto: ML Emozioni - Fase 2 (Late Fusion)

Descrizione:
Carica 3 segnali (Affect, ECG, EDA), li normalizza separatamente,
ma invece di fonderli, li restituisce come 3 tensori indipendenti di forma (1, 1000).
=====================================================================================
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from config_FASE2_late import SPLIT_INDEX_PATH, WINDOW_SIZE # Assicurati che il nome del config sia giusto!

class PopaneDatasetLateFusion(Dataset):
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

        # 1. Lettura a 3 canali dal CSV
        # Colonne: 1=affect, 2=ECG, 3=EDA
        chunk = pd.read_csv(
            file_path, 
            skiprows=start_row + 1, 
            nrows=self.window_size, 
            header=None,
            usecols=[1, 2, 3] 
        )
        
        # 2. Estrazione e Trasposta (forma: 3 x 1000)
        raw_signals = chunk.values.T.astype(np.float32)
        raw_signals = np.nan_to_num(raw_signals)
        
        # 3. Normalizzazione Indipendente (Z-Score)
        means = raw_signals.mean(axis=1, keepdims=True)
        stds = raw_signals.std(axis=1, keepdims=True) + 1e-8
        norm_signals = (raw_signals - means) / stds

        # 4. IL TRUCCO DELLA LATE FUSION: SEPARIAMO I SEGNALI!
        # norm_signals[0] è l'Affect, norm_signals[1] è l'ECG, norm_signals[2] è l'EDA
        # Usiamo np.expand_dims per fargli avere la forma (1, 1000) richiesta dalla CNN
        tensor_affect = torch.FloatTensor(np.expand_dims(norm_signals[0], axis=0))
        tensor_ecg = torch.FloatTensor(np.expand_dims(norm_signals[1], axis=0))
        tensor_eda = torch.FloatTensor(np.expand_dims(norm_signals[2], axis=0))
        
        label_tensor = torch.tensor(label, dtype=torch.float32)

        # Restituiamo 4 cose invece di 2!
        return tensor_affect, tensor_ecg, tensor_eda, label_tensor