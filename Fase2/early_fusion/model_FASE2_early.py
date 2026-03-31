"""
=====================================================================================
Module: models_fase2.py
Project: ML Emotions - Phase 2 (Multimodal Early Fusion)

Description:
Multimodal 1D-CNN Architecture.
Takes as input a tensor with 3 channels (Affect, ECG, EDA) simultaneously.
Uses Batch Normalization to stabilize calculations between different signals.
=====================================================================================
"""

import torch
import torch.nn as nn
from config_FASE2_early import WINDOW_SIZE

class MultimodalEarlyFusionCNN(nn.Module):
    """1D CNN early-fusion for three signals (Affect, ECG, EDA)."""

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
        """Computes the output length after Conv1D + Pooling without dynamic padding."""
        length = window_size
        for _ in range(conv_blocks):
            length = length - (kernel_size - 1)
            if length <= 0:
                raise ValueError(f"The window_size ({window_size}) is too small for the convolutional pipeline")
            length = length // pool_size
        if length <= 0:
            raise ValueError(f"The window_size ({window_size}) should be reduced or increased to maintain positive dimensions")
        return length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Outputs logits for binary classification with BCEWithLogitsLoss."""
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted probabilities after sigmoid."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def num_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
