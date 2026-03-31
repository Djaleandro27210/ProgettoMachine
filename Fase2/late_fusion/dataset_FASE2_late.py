"""
=====================================================================================
Module: dataset_FASE2_late.py
Project: ML Emotions - Phase 2 (Late Fusion)

Description:
Loads 3 signals (Affect, ECG, EDA), normalizes them separately,
but instead of fusing them, returns them as 3 independent tensors of shape (1, 1000).
=====================================================================================
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from config_FASE2_late import SPLIT_INDEX_PATH, WINDOW_SIZE

class PopaneDatasetLateFusion(Dataset):
    def __init__(self, split_type="train", transform=None):
        self.full_index = pd.read_csv(SPLIT_INDEX_PATH)
        self.index_df = self.full_index[self.full_index['split'] == split_type].reset_index(drop=True)
        self.window_size = WINDOW_SIZE
        self.transform = transform

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        row_info = self.index_df.iloc[idx]
        signals = self._read_window(row_info['file_path'], row_info['start_row'])
        signals = self._normalize(signals)

        if self.transform is not None:
            signals = self.transform(signals)

        tensor_channels = [
            torch.from_numpy(channel).unsqueeze(0).to(torch.float32)
            for channel in signals
        ]

        label_tensor = torch.tensor(row_info['label'], dtype=torch.float32)

        # Output: affect, ecg, eda, label
        return (*tensor_channels, label_tensor)

    def _read_window(self, file_path: str, start_row: int) -> np.ndarray:
        """Reads a window of rows from the CSV, returns array 3xwindow_size."""
        chunk = pd.read_csv(
            file_path,
            skiprows=start_row + 1,
            nrows=self.window_size,
            header=None,
            usecols=[1, 2, 3],
        )

        signals = chunk.values.T.astype(np.float32)
        return np.nan_to_num(signals)

    def _normalize(self, signals: np.ndarray) -> np.ndarray:
        """Applies z-score normalization on each channel separately."""
        means = signals.mean(axis=1, keepdims=True)
        stds = signals.std(axis=1, keepdims=True)
        stds = np.where(stds < 1e-8, 1e-8, stds)
        return (signals - means) / stds