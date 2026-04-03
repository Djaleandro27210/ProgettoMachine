"""
dataset_FASE1.py
PyTorch Dataset for Phase 1 (ECG). Handles lazy loading, windowing, and Z-score normalization.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple

from config_FASE1 import SPLIT_INDEX_PATH, WINDOW_SIZE


class PopaneDataset(Dataset):
    """PyTorch Dataset for loading and preprocessing ECG time-series data."""
    
    ECG_COLUMN_INDEX = 2
    HEADER_OFFSET = 1
    EPSILON = 1e-8

    def __init__(self, split_type: str = "train") -> None:
        self.split_type = split_type
        self.window_size = WINDOW_SIZE
        
        full_index = pd.read_csv(SPLIT_INDEX_PATH)
        self.index_df = full_index[full_index['split'] == self.split_type].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.index_df)

    def _load_and_preprocess_signal(self, file_path: str, start_row: int) -> np.ndarray:
        # Lazy loading: extract only the required chunk
        chunk = pd.read_csv(
            file_path, 
            skiprows=start_row + self.HEADER_OFFSET, 
            nrows=self.window_size, 
            header=None,
            usecols=[self.ECG_COLUMN_INDEX] 
        )
        
        # Flatten, force numeric conversion, and handle NaNs
        ecg_signal = pd.to_numeric(chunk.values.flatten(), errors='coerce')
        ecg_signal = np.nan_to_num(ecg_signal).astype(np.float32)
        
        # Z-score normalization
        mean = ecg_signal.mean()
        std = ecg_signal.std() + self.EPSILON 
        
        return (ecg_signal - mean) / std

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row_info = self.index_df.iloc[idx]

        ecg_signal = self._load_and_preprocess_signal(
            row_info['file_path'], 
            row_info['start_row']
        )

        ecg_tensor = torch.FloatTensor(ecg_signal).unsqueeze(0)
        label_tensor = torch.tensor(row_info['label'], dtype=torch.float32)

        return ecg_tensor, label_tensor