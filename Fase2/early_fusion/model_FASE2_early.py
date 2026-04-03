"""
model_FASE2_early.py
Multimodal 1D-CNN architecture for Early Fusion.
Processes a 3-channel input tensor (Affect, ECG, EDA) simultaneously.
"""

import torch
import torch.nn as nn
from config_FASE2_early import WINDOW_SIZE


class MultimodalEarlyFusionCNN(nn.Module):
    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        super().__init__()
        self.window_size = window_size
        
        self.feature_extractor = self._build_feature_extractor()
        self.flatten_size = self._calculate_flatten_size()
        self.classifier = self._build_classifier()

    def _build_feature_extractor(self) -> nn.Sequential:
        return nn.Sequential(
            # First convolutional block: 3 input channels (Affect, ECG, EDA)
            nn.Conv1d(in_channels=3, out_channels=18, kernel_size=7),
            nn.BatchNorm1d(num_features=18),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            # Second convolutional block
            nn.Conv1d(in_channels=18, out_channels=18, kernel_size=7),
            nn.BatchNorm1d(num_features=18),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

    def _calculate_flatten_size(self) -> int:
        # Pass a 3-channel dummy tensor to dynamically compute the flatten size
        dummy_input = torch.zeros(1, 3, self.window_size)
        
        with torch.no_grad():
            dummy_output = self.feature_extractor(dummy_input)
            
        return dummy_output.view(1, -1).size(1)

    def _build_classifier(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_features=self.flatten_size, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=64, out_features=1) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)