"""
=====================================================================================
Modulo: dataset.py
Progetto: ML Emozioni (Popane Dataset) - Fase 1 (ECG)

Descrizione:
Questo script è il "Motore di Caricamento Dati" (Data Pipeline) per PyTorch.
Definisce la classe custom `PopaneDataset`, che fa da ponte tra i file CSV 
pesantissimi sul disco rigido e la rete neurale (1D-CNN), gestendo la memoria.

Funzionamento (Cosa succede quando la rete chiede un batch di dati):
1. LAZY LOADING: Legge l'indice (dataset_index_split.csv), apre il file raw specifico
   ed estrae SOLO le righe richieste (Windowing), evitando di saturare la RAM.
2. ESTRAZIONE: Seleziona solo la colonna del segnale ECG.
3. PREPROCESSING: Applica la normalizzazione Z-Score ((x - mean) / std) per 
   centrare il segnale, aiutando la rete neurale a convergere più velocemente.
4. TENSORIZZAZIONE: Converte gli array in Tensori PyTorch, impostando la forma 
   (1, WINDOW_SIZE) richiesta dalle reti convoluzionali 1D (Canali, Lunghezza).
=====================================================================================
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# ==========================================
# IMPORTIAMO LE COSTANTI DAL CONFIG
# ==========================================

from config_FASE1 import SPLIT_INDEX_PATH, WINDOW_SIZE

class PopaneDataset(Dataset):
    def __init__(self, split_type="train"):
        """
        Args:
            split_type: 'train', 'val' o 'test' per caricare solo i dati giusti.
                        (index_path e window_size ora vengono presi da config.py)
        """
        # 1. Carichiamo l'indice generale usando il percorso dal config
        self.full_index = pd.read_csv(SPLIT_INDEX_PATH)
        
        # 2. Filtriamo SOLO per lo split richiesto e resettiamo l'indice
        self.index_df = self.full_index[self.full_index['split'] == split_type].reset_index(drop=True)
        
        # Salviamo la window size dal config per usarla nella lettura
        self.window_size = WINDOW_SIZE

    def __len__(self):
        # PyTorch ha bisogno di sapere quanti esempi ci sono in questo set
        return len(self.index_df)

    def __getitem__(self, idx):
        # 1. Recuperiamo le coordinate della fettina specifica
        row_info = self.index_df.iloc[idx]
        file_path = row_info['file_path']
        start_row = row_info['start_row']
        label = row_info['label']

        # 2. Caricamento Pigro
        # AGGIUNTA VIP: + 1 a start_row per saltare l'intestazione testuale (timestamp, affect, ECG...)
        chunk = pd.read_csv(
            file_path, 
            skiprows=start_row + 1, 
            nrows=self.window_size, 
            header=None,
            usecols=[2] 
        )
        
        # Trasformiamo la colonna in un array 1D e FORZIAMO il tipo a numero decimale (float32)
        # Se c'è sporcizia nel file, lo convertiamo a numero in modo sicuro
     # Convertiamo in numerico, forzando gli errori a diventare NaN, e poi riempiamo i NaN con zero
        ecg_signal = pd.to_numeric(chunk.values.flatten(), errors='coerce')
        ecg_signal = np.nan_to_num(ecg_signal).astype(np.float32)
      # --- AGGIUNGI QUESTE RIGHE PER NORMALIZZARE ---
        mean = ecg_signal.mean()
        std = ecg_signal.std() + 1e-8 # Aggiungiamo un numero minuscolo per non dividere per zero
        ecg_signal = (ecg_signal - mean) / std
        # ----------------------------------------------

        # Trasforma in tensore
        ecg_tensor = torch.FloatTensor(ecg_signal).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return ecg_tensor, label_tensor