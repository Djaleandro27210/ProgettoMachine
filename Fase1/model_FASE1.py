"""
1D-CNN model architecture for Phase 1 (ECG-only emotion classification).

This module defines a 1D convolutional neural network designed for binary
emotion classification from ECG time-series data. The architecture consists
of two convolutional blocks with batch normalization and pooling, followed
by a fully-connected classifier.

Architecture:
- Conv1D (1->18 filters, kernel=7) + BatchNorm + ReLU + MaxPool
- Conv1D (18->18 filters, kernel=7) + BatchNorm + ReLU + MaxPool
- Flatten + FC(flatten_size->64) + ReLU + Dropout(0.4) + FC(64->1)
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn

from config_FASE1 import WINDOW_SIZE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model hyperparameters
CONV1_OUT_CHANNELS = 18
CONV2_OUT_CHANNELS = 18
CONV_KERNEL_SIZE = 7
POOL_KERNEL_SIZE = 2
FC_HIDDEN_DIM = 64
DROPOUT_RATE = 0.4
OUTPUT_DIM = 1


class Emotion1DCNN(nn.Module):
    """1D-CNN for binary emotion classification from ECG signals.
    
    Input: (batch_size, 1, WINDOW_SIZE) - single ECG channel
    Output: (batch_size, 1) - logit for binary classification
    
    Architecture:
    - 2 convolutional blocks with batch norm, ReLU, and max pooling
    - Fully connected layers with dropout
    - Single output neuron (binary classification with sigmoid during inference)
    """
    
    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        """Initialize the 1D-CNN model.
        
        Args:
            window_size: Length of input ECG time-series.
                        Default: WINDOW_SIZE from config.
        """
        super(Emotion1DCNN, self).__init__()
        self.window_size = window_size
        
        # Feature extraction blocks
        self.feature_extractor = self._build_feature_extractor()

        # Compute flattened dimension after convolutions
        self.flatten_size = self._compute_flatten_size()

        # Classification head
        self.classifier = self._build_classifier()

        # Initialize model weights
        self._initialize_weights()

        logger.info(
            f"Emotion1DCNN initialized: "
            f"window_size={window_size}, flatten_size={self.flatten_size}"
        )
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Build a single convolutional block with normalization and pooling."""
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=CONV_KERNEL_SIZE),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=POOL_KERNEL_SIZE),
        )

    def _build_feature_extractor(self) -> nn.Sequential:
        """Build the convolutional feature extraction layers."""
        return nn.Sequential(
            self._conv_block(1, CONV1_OUT_CHANNELS),
            self._conv_block(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS),
        )
    
    def _compute_flatten_size(self) -> int:
        """Dynamically compute the flattened size after convolutions.
        
        This is robust to changes in WINDOW_SIZE or architecture.
        
        Returns:
            Number of features after flattening conv output.
        """
        dummy_input = torch.zeros(1, 1, self.window_size)
        dummy_output = self.feature_extractor(dummy_input)
        flatten_size = dummy_output.view(1, -1).size(1)
        logger.debug(f"Computed flatten_size: {flatten_size}")
        return flatten_size
    
    def _build_classifier(self) -> nn.Sequential:
        """Build the fully-connected classification layers."""
        return nn.Sequential(
            nn.Linear(self.flatten_size, FC_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(FC_HIDDEN_DIM, OUTPUT_DIM),
        )

    def _initialize_weights(self) -> None:
        """Initialize weights with Kaiming normalization and constant biases."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 1, window_size).
            
        Returns:
            Output logits of shape (batch_size, 1).
        """
        # Feature extraction
        x = self.feature_extractor(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x