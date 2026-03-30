"""
Data Pipeline for Phase 2 Early Fusion (3 modalities: Affect, ECG, EDA).

This module provides lazy loading and independent Z-score normalization
for three time-series signals loaded simultaneously. Each signal is
normalized independently to prevent signal scaling dominance issues.

Key Features:
- Multi-modal loading: Affect (column 1), ECG (column 2), EDA (column 3)
- Per-channel normalization: Z-score computed independently per signal
- PyTorch integration: outputs (channels=3, sequence_length) format
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config_FASE2_early import SPLIT_INDEX_PATH, WINDOW_SIZE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dataset constants
SIGNAL_COLUMNS = [1, 2, 3]  # Affect, ECG, EDA column indices
NUM_CHANNELS = 3
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


def load_multimodal_window(
    file_path: str,
    start_row: int,
    window_size: int,
    column_indices: list = SIGNAL_COLUMNS,
    skip_rows: int = HEADER_SKIP_ROWS,
) -> np.ndarray:
    """Load three signals from CSV file.
    
    Args:
        file_path: Path to the raw CSV file.
        start_row: Starting row index (0-based) within the file.
        window_size: Number of rows to read.
        column_indices: List of column indices for signals (default: [1, 2, 3]).
        skip_rows: Rows to skip before start_row.
        
    Returns:
        2D numpy array of shape (num_channels, window_size) in float32.
    """
    chunk = pd.read_csv(
        file_path,
        skiprows=start_row + skip_rows,
        nrows=window_size,
        header=None,
        usecols=column_indices,
    )
    
    # Transpose to get (channels, time_steps)
    raw_signals = chunk.values.T.astype(np.float32)
    raw_signals = np.nan_to_num(raw_signals)
    
    return raw_signals


def normalize_multimodal_signals(
    signals: np.ndarray,
    epsilon: float = NORMALIZE_EPSILON,
) -> np.ndarray:
    """Apply independent Z-score normalization per channel.
    
    Each signal channel is normalized using its own mean and std to prevent
    scaling dominance (e.g., EDA voltage overwhelming ECG millivolts).
    
    Args:
        signals: Input array of shape (num_channels, window_size).
        epsilon: Small value to prevent division by zero.
        
    Returns:
        Normalized signals with shape (num_channels, window_size).
    """
    means = signals.mean(axis=1, keepdims=True)
    stds = signals.std(axis=1, keepdims=True) + epsilon
    normalized = (signals - means) / stds
    return normalized.astype(np.float32)


def signals_to_tensor(
    signals: np.ndarray,
    label: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert signals and label to PyTorch tensors.
    
    Args:
        signals: 2D array of shape (num_channels, window_size).
        label: Integer label (0 or 1).
        
    Returns:
        Tuple of (signals_tensor, label_tensor).
    """
    signals_tensor = torch.FloatTensor(signals)
    label_tensor = torch.tensor(label, dtype=torch.float32)
    return signals_tensor, label_tensor


class PopaneDatasetMultimodal(Dataset):
    """PyTorch Dataset for multimodal time-series (Affect, ECG, EDA).
    
    Features:
    - Lazy loading: reads only required windows from disk
    - Multi-channel: simultaneous loading of 3 signals
    - Independent normalization: per-channel Z-score to prevent scaling dominance
    - Subject-level split: consistent train/val/test assignment
    """
    
    def __init__(
        self,
        split_type: str = "train",
        index_path: str = SPLIT_INDEX_PATH,
        window_size: int = WINDOW_SIZE,
    ):
        """Initialize multimodal dataset.
        
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
        
        logger.info(
            f"PopaneDatasetMultimodal initialized for '{split_type}' split "
            f"with {len(self.index_df)} samples"
        )
    
    def __len__(self) -> int:
        """Return total number of samples in this split."""
        return len(self.index_df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess a single multimodal sample.
        
        Args:
            idx: Sample index within the split.
            
        Returns:
            Tuple of (signals_tensor, label_tensor) with shapes:
            - signals_tensor: (3, window_size)
            - label_tensor: scalar
        """
        row_info = self.index_df.iloc[idx]
        file_path = row_info["file_path"]
        start_row = row_info["start_row"]
        label = int(row_info["label"])
        
        signals = load_multimodal_window(file_path, start_row, self.window_size)
        signals = normalize_multimodal_signals(signals)
        signals_tensor, label_tensor = signals_to_tensor(signals, label)
        
        return signals_tensor, label_tensor


if __name__ == "__main__":
    # Quick sanity check
    logger.info("Running dataset sanity check...")
    test_ds = PopaneDatasetMultimodal(split_type="train")
    if len(test_ds) > 0:
        signals, label = test_ds[0]
        logger.info(f"Sample signals shape: {signals.shape} (expected: ({NUM_CHANNELS}, {WINDOW_SIZE}))")
        logger.info(f"Sample label: {label}")
    else:
        logger.warning("Train set is empty")