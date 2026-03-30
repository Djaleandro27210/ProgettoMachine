"""
Data Pipeline for Phase 1 (ECG-only) - PopaneDataset PyTorch Module

This module provides lazy loading and preprocessing for ECG time-series data
from CSV files. The PopaneDataset class is designed to efficiently handle
large datasets without saturating RAM by loading only requested windows.

Key Features:
- Lazy loading: reads only requested rows from disk
- Z-score normalization: centers and scales signals
- PyTorch integration: outputs (channels, length) tensor format for 1D-CNN
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config_FASE1 import SPLIT_INDEX_PATH, WINDOW_SIZE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset constants
ECG_COLUMN_INDEX = 2
HEADER_SKIP_ROWS = 1
NORMALIZE_EPSILON = 1e-8
VALID_SPLITS = {"train", "val", "test"}


def load_split_index(index_path: str) -> pd.DataFrame:
    """Load the split index CSV file.
    
    Args:
        index_path: Path to the split index CSV.
        
    Returns:
        DataFrame with columns: file_path, subject_id, emotion, label, start_row, end_row, split.
    """
    logger.info(f"Loading split index from {index_path}")
    df = pd.read_csv(index_path)
    logger.info(f"Loaded {len(df)} total samples")
    return df


def filter_by_split(df: pd.DataFrame, split_type: str) -> pd.DataFrame:
    """Filter index by split type and reset index.
    
    Args:
        df: Full index DataFrame.
        split_type: 'train', 'val', or 'test'.
        
    Returns:
        Filtered DataFrame with rows belonging to split_type.
        
    Raises:
        ValueError: If split_type is invalid.
    """
    if split_type not in VALID_SPLITS:
        raise ValueError(f"split_type must be one of {VALID_SPLITS}, got '{split_type}'")
    
    filtered = df[df["split"] == split_type].reset_index(drop=True)
    logger.info(f"Split '{split_type}': {len(filtered)} samples")
    return filtered


def load_ecg_window(
    file_path: str,
    start_row: int,
    window_size: int,
    column_index: int = ECG_COLUMN_INDEX,
    skip_rows: int = HEADER_SKIP_ROWS,
) -> np.ndarray:
    """Load a single ECG window from CSV file.
    
    Args:
        file_path: Path to the raw CSV file.
        start_row: Starting row index (0-based) within the file.
        window_size: Number of rows to read.
        column_index: Column index for ECG signal (default: 2).
        skip_rows: Rows to skip before start_row.
        
    Returns:
        1D numpy array of float32 values.
    """
    chunk = pd.read_csv(
        file_path,
        skiprows=start_row + skip_rows,
        nrows=window_size,
        header=None,
        usecols=[column_index],
    )
    
    ecg_array = chunk.values.flatten()
    ecg_signal = pd.to_numeric(ecg_array, errors="coerce")
    ecg_signal = np.nan_to_num(ecg_signal).astype(np.float32)
    
    return ecg_signal


def normalize_signal(signal: np.ndarray, epsilon: float = NORMALIZE_EPSILON) -> np.ndarray:
    """Apply Z-score normalization to signal.
    
    Args:
        signal: Input signal (1D array).
        epsilon: Small value to prevent division by zero.
        
    Returns:
        Normalized signal (mean=0, std~1).
    """
    mean = signal.mean()
    std = signal.std() + epsilon
    normalized = (signal - mean) / std
    return normalized.astype(np.float32)


def signal_to_tensor(signal: np.ndarray, label: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert signal and label to PyTorch tensors.
    
    Args:
        signal: 1D numpy array of shape (window_size,).
        label: Integer label (0 or 1).
        
    Returns:
        Tuple of (signal_tensor, label_tensor) where signal_tensor has shape (1, window_size).
    """
    signal_tensor = torch.FloatTensor(signal).unsqueeze(0)  # Add channel dimension
    label_tensor = torch.tensor(label, dtype=torch.float32)
    return signal_tensor, label_tensor


class PopaneDataset(Dataset):
    """PyTorch Dataset for ECG time-series from Popane dataset.
    
    Features:
    - Lazy loading: reads only required windows from disk
    - Subject-level split: subjects are assigned to train/val/test, not samples
    - Z-score normalization: centers and scales each window independently
    - 1D-CNN format: outputs (channels=1, sequence_length) tensors
    """
    
    def __init__(
        self,
        split_type: str = "train",
        index_path: str = SPLIT_INDEX_PATH,
        window_size: int = WINDOW_SIZE,
    ):
        """Initialize dataset.
        
        Args:
            split_type: One of 'train', 'val', 'test'.
            index_path: Path to split_index CSV file.
            window_size: Number of samples per window.
            
        Raises:
            ValueError: If split_type is invalid or index file not found.
        """
        self.split_type = split_type
        self.window_size = window_size
        
        full_index = load_split_index(index_path)
        self.index_df = filter_by_split(full_index, split_type)
        
        logger.info(f"PopaneDataset initialized for '{split_type}' split with {len(self.index_df)} samples")
    
    def __len__(self) -> int:
        """Return total number of samples in this split."""
        return len(self.index_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess a single sample.
        
        Args:
            idx: Sample index within the split.
            
        Returns:
            Tuple of (ecg_tensor, label_tensor) ready for training.
        """
        row_info = self.index_df.iloc[idx]
        file_path = row_info["file_path"]
        start_row = row_info["start_row"]
        label = int(row_info["label"])
        
        ecg_signal = load_ecg_window(file_path, start_row, self.window_size)
        ecg_signal = normalize_signal(ecg_signal)
        signal_tensor, label_tensor = signal_to_tensor(ecg_signal, label)
        
        return signal_tensor, label_tensor