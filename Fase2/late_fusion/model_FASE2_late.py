"""
=====================================================================================
Modulo: model_fase2_late.py
Progetto: ML Emozioni - Fase 2 (Late Fusion)

Descrizione:
Architettura 1D-CNN Unimodale (a 1 singolo canale).
Verrà istanziata 3 volte (una per l'ECG, una per l'EDA, una per l'Affect).
=====================================================================================
"""
import torch
import torch.nn as nn
from config_FASE2_late import WINDOW_SIZE # Assicurati che il nome del config sia giusto

class UnimodalCNN(nn.Module):
    def __init__(self, window_size=WINDOW_SIZE):
        super(UnimodalCNN, self).__init__()
        
        # Lavoriamo con 1 solo canale in ingresso!
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=18, kernel_size=7),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=18, out_channels=18, kernel_size=7),
            nn.BatchNorm1d(18),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        dummy_x = torch.zeros(1, 1, window_size)
        dummy_out = self.feature_extractor(dummy_x)
        self.flatten_size = dummy_out.view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x