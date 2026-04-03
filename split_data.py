"""
split_data.py
Performs subject-wise dataset splitting (70% Train, 15% Val, 15% Test).
Ensures no data leakage between splits and validates undersampling balance.
"""

import pandas as pd
import numpy as np

from config import INDEX_PATH, SPLIT_INDEX_PATH


def perform_subject_split() -> None:
    # Set fixed seed for scientific reproducibility
    np.random.seed(42)

    print(f"[INFO] Loading dataset index from: {INDEX_PATH}")
    df = pd.read_csv(INDEX_PATH)

    unique_subjects = df['subject_id'].unique()
    print(f"[INFO] Found {len(unique_subjects)} unique subjects.")

    # Randomize subject order
    np.random.shuffle(unique_subjects)

    # Calculate split quotas (70% / 15% / 15%)
    total_subjects = len(unique_subjects)
    n_train = int(total_subjects * 0.70)
    n_val = int(total_subjects * 0.15)

    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:n_train + n_val]
    test_subjects = unique_subjects[n_train + n_val:]

    print(f"[INFO] Subject Distribution -> Train: {len(train_subjects)} | Val: {len(val_subjects)} | Test: {len(test_subjects)}")

    # Create a mapping dictionary for efficient assignment
    print("[INFO] Assigning windows to target splits...")
    split_map = {}
    for subject in train_subjects:
        split_map[subject] = 'train'
    for subject in val_subjects:
        split_map[subject] = 'val'
    for subject in test_subjects:
        split_map[subject] = 'test'

    # Apply mapping to create the 'split' column
    df['split'] = df['subject_id'].map(split_map)

    # Save to disk
    df.to_csv(SPLIT_INDEX_PATH, index=False)
    print(f"[INFO] Split index successfully saved to: {SPLIT_INDEX_PATH}\n")

    # =====================================================================
    # POST-SPLIT STATISTICAL REPORT
    # =====================================================================
    print("[STAT] FINAL DATASET DISTRIBUTION REPORT (Undersampling Validation):")
    for split_name in ['train', 'val', 'test']:
        subset = df[df['split'] == split_name]
        total = len(subset)
        
        neg = len(subset[subset['label'] == 0])
        pos = len(subset[subset['label'] == 1])
        
        perc_neg = (neg / total) * 100 if total > 0 else 0
        perc_pos = (pos / total) * 100 if total > 0 else 0
        
        print(f"[{split_name.upper()}] Total Windows: {total}")
        print(f"    -> Negative (0): {neg:<5} ({perc_neg:.1f}%)")
        print(f"    -> Positive (1): {pos:<5} ({perc_pos:.1f}%)\n")


if __name__ == "__main__":
    perform_subject_split()