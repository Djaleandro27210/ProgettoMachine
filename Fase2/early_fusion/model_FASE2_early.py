"""
=====================================================================================
Modulo: models_fase2.py
Progetto: ML Emozioni - Fase 2 (Multimodale Early Fusion)

Descrizione:
Architettura 1D-CNN Multimodale. 
Prende in ingresso un tensore con 3 canali (Affect, ECG, EDA) contemporaneamente.
Usa la Batch Normalization per stabilizzare i calcoli tra i diversi segnali.
=====================================================================================
"""

import torch
import torch.nn as nn
from config_FASE2_early import WINDOW_SIZE

class MultimodalEarlyFusionCNN(nn.Module):
    def __init__(self, window_size=WINDOW_SIZE):
        super(MultimodalEarlyFusionCNN, self).__init__()
        
        # ==========================================
        # 1. MODULO DI ESTRAZIONE FEATURE (A 3 CANALI)
        # ==========================================
        self.feature_extractor = nn.Sequential(
            # LA MAGIA È QUI: in_channels=3 (Affect, ECG, EDA)
            nn.Conv1d(in_channels=3, out_channels=18, kernel_size=7),
            nn.BatchNorm1d(18), # Stabilizza i segnali fusi
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=18, out_channels=18, kernel_size=7),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # ==========================================
        # 2. CALCOLO AUTOMATICO FLATTEN
        # ==========================================
        # Simuliamo un passaggio con un tensore (Batch=1, Canali=3, Lunghezza=1000)
        dummy_x = torch.zeros(1, 3, window_size)
        dummy_out = self.feature_extractor(dummy_x)
        self.flatten_size = dummy_out.view(1, -1).size(1)
        
        # ==========================================
        # 3. CLASSIFICATORE FINALE
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout ridotto per non frenare troppo l'apprendimento
            nn.Linear(64, 1) # Output binario
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x