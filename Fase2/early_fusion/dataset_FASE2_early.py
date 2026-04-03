"""
dataset_FASE2_early.py
PyTorch Dataset for Phase 2 (Early Fusion). 
Loads 3 signals (Affect, ECG, EDA) and applies independent Z-score normalization.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple

from config_FASE2_early import SPLIT_INDEX_PATH, WINDOW_SIZE


class PopaneDatasetMultimodal(Dataset):
    
    # Column indices in the raw CSV: 1=Affect, 2=ECG, 3=EDA
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
        
        # Transpose to format (channels, samples) -> e.g., (3, 1000)
        raw_signals = chunk.values.T.astype(np.float32)
        raw_signals = np.nan_to_num(raw_signals)
        
        # Independent Z-score normalization per channel (axis=1)
        means = raw_signals.mean(axis=1, keepdims=True)
        stds = raw_signals.std(axis=1, keepdims=True) + self.EPSILON
        
        return (raw_signals - means) / stds

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row_info = self.index_df.iloc[idx]

        normalized_signals = self._load_and_preprocess_signals(
            row_info['file_path'], 
            row_info['start_row']
        )

        # Output shape is already (3, WINDOW_SIZE), so no unsqueeze is needed
        signals_tensor = torch.FloatTensor(normalized_signals)
        label_tensor = torch.tensor(row_info['label'], dtype=torch.float32)

        return signals_tensor, label_tensor


# Simple execution test
if __name__ == "__main__":
    test_ds = PopaneDatasetMultimodal(split_type="train")
    signals, label = test_ds[0]
    
    print(f"[INFO] Signal Tensor Shape: {signals.shape} (Expected: 3, {WINDOW_SIZE})")
    print(f"[INFO] Label: {label}")