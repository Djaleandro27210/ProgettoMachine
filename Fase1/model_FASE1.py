"""
=====================================================================================
Modulo: models.py
Progetto: ML Emozioni (Popane Dataset) - Fase 1 (ECG)

Descrizione:
Definizione dell'architettura della Rete Neurale 1D-CNN, ispirata al paper di 
riferimento. Prende in input l'ECG (1 canale, lunghezza variabile) e restituisce
una probabilità (Positivo vs Negativo).
=====================================================================================
"""

import torch
import torch.nn as nn
from config_FASE1 import WINDOW_SIZE

class Emotion1DCNN(nn.Module):
    def __init__(self, window_size=WINDOW_SIZE):
        super(Emotion1DCNN, self).__init__()
        
        # ==========================================
        # 1. MODULO DI FEATURE EXTRACTION (Convoluzioni)
        # ==========================================
        self.feature_extractor = nn.Sequential(
            # C1: 18 filtri (kernel_size=7 come da paper) --> Output: (124)

            nn.Conv1d(in_channels=1, out_channels=18, kernel_size=7),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            # S1: Down-sampling (Pooling)-->output: (62)
            nn.MaxPool1d(kernel_size=2),
            
            # C2: 18 filtri (kernel_size=7) --> Output: (56)
            nn.Conv1d(in_channels=18, out_channels=18, kernel_size=7),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            # S2: Down-sampling (Pooling)-->output: (28)
            nn.MaxPool1d(kernel_size=2)
        )
        
        # ==========================================
        # 2. CALCOLO AUTOMATICO DIMENSIONE FLATTEN
        # ==========================================
        # Facciamo passare un "tensore finto" di zeri attraverso i layer appena creati
        # per vedere la dimensione esatta in uscita. (A prova di bomba se cambi WINDOW_SIZE)
        dummy_x = torch.zeros(1, 1, window_size)
        dummy_out = self.feature_extractor(dummy_x)
        self.flatten_size = dummy_out.view(1, -1).size(1)
        
        # ==========================================
        # 3. MODULO DI CLASSIFICAZIONE (Fully Connected)
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3), # Dropout al 30% per evitare l'Overfitting!
            nn.Linear(64, 1) # Output: 1 singolo neurone (per classificazione Binaria)
        )

    def forward(self, x):
        # Passaggio 1: Estrazione feature
        x = self.feature_extractor(x)
        
        # Passaggio 2: Flatten (appiattiamo la matrice in un vettore 1D)
        x = torch.flatten(x, 1)
        
        # Passaggio 3: Classificazione finale
        x = self.classifier(x)
        return x