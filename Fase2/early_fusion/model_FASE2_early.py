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
    """1D CNN early-fusion per tre segnali (Affect, ECG, EDA)."""

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        in_channels: int = 3,
        hidden_channels: tuple[int, int] = (18, 18),
        kernel_size: int = 7,
        fc_dim: int = 64,
        dropout_prob: float = 0.5,
    ):
        super().__init__()

        self.window_size = window_size
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels[0], kernel_size=kernel_size),
            nn.BatchNorm1d(hidden_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=hidden_channels[0], out_channels=hidden_channels[1], kernel_size=kernel_size),
            nn.BatchNorm1d(hidden_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.flatten_size = hidden_channels[1] * self._compute_feature_map_length(window_size)

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_size, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(fc_dim, 1),
        )

    @staticmethod
    def _compute_feature_map_length(window_size: int, kernel_size: int = 7, pool_size: int = 2, conv_blocks: int = 2) -> int:
        """Calcola la lunghezza di uscita dopo Conv1D + Pooling senza tangenti dinamiche."""
        length = window_size
        for _ in range(conv_blocks):
            length = length - (kernel_size - 1)
            if length <= 0:
                raise ValueError(f"Il window_size ({window_size}) è troppo piccolo per la pipeline convoluzionale")
            length = length // pool_size
        if length <= 0:
            raise ValueError(f"Il window_size ({window_size}) va ridotto o aumentato per mantenere dimensioni positive")
        return length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Output logits (per binarizzazione BCEWithLogitsLoss)."""
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Restituisce le probabilità predette dopo sigmoid."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def num_parameters(self) -> int:
        """Restituisce il numero totale di parametri addestrabili."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
