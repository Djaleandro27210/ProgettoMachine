"""
dataset_FASE3_late.py
PyTorch Dataset for Phase 3 (Late Fusion).
Loads Affect, ECG, and EDA signals, normalizes them independently, 
and returns them as three separate tensors.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple

from config_FASE2_late import SPLIT_INDEX_PATH, WINDOW_SIZE


class PopaneDatasetLateFusion(Dataset):
    
    # Column indices: 1=Affect, 2=ECG, 3=EDA
    TARGET_COLS = [1, 2, 3]  
    HEADER_OFFSET = 1
    EPSILON = 1e-8

    def __init__(self, split_type: str = "train") -> None:
        self.split_type = split_type
        self.window_size = WINDOW_SIZE
        
        full_index = pd.read_csv(SPLIT_INDEX_PATH)
        self.index_df = full_index[full_index['split'] == self.split_type].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.index_df)

    def _load_and_preprocess_signals(self, file_path: str, start_row: int) -> np.ndarray:
        chunk = pd.read_csv(
            file_path, 
            skiprows=start_row + self.HEADER_OFFSET, 
            nrows=self.window_size, 
            header=None,
            usecols=self.TARGET_COLS 
        )
        
        raw_signals = chunk.values.T.astype(np.float32)
        raw_signals = np.nan_to_num(raw_signals)
        
        # Independent Z-score normalization per channel
        means = raw_signals.mean(axis=1, keepdims=True)
        stds = raw_signals.std(axis=1, keepdims=True) + self.EPSILON
        
        return (raw_signals - means) / stds

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row_info = self.index_df.iloc[idx]

        norm_signals = self._load_and_preprocess_signals(
            row_info['file_path'], 
            row_info['start_row']
        )

        # Separate the channels for Late Fusion. Shape becomes (1, WINDOW_SIZE) per tensor.
        tensor_affect = torch.FloatTensor(norm_signals[0]).unsqueeze(0)
        tensor_ecg = torch.FloatTensor(norm_signals[1]).unsqueeze(0)
        tensor_eda = torch.FloatTensor(norm_signals[2]).unsqueeze(0)
        
        label_tensor = torch.tensor(row_info['label'], dtype=torch.float32)

        return tensor_affect, tensor_ecg, tensor_eda, label_tensor