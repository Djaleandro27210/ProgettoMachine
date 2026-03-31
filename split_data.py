"""
Module: split_data.py
Project: ML Emotions - Dataset Splitting

Description:
This module performs subject-wise splitting of the dataset into train, validation, and test sets.
It ensures that data from the same subject does not appear in multiple splits to prevent data leakage.
"""

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

from config import INDEX_PATH, SPLIT_INDEX_PATH


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ["train", "val", "test"]


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float
    test: float

    def __post_init__(self):
        if not (0.99 < sum([self.train, self.val, self.test]) < 1.01):
            raise ValueError("Split ratios must sum to 1.0")


def load_index(index_path: str) -> pd.DataFrame:
    logger.info("Loading index from %s", index_path)
    df = pd.read_csv(index_path)
    logger.info("Index loaded: %d rows", len(df))
    return df


def extract_subjects(df: pd.DataFrame) -> np.ndarray:
    subjects = df["subject_id"].unique()
    logger.info("Found %d unique subjects", len(subjects))
    return subjects


def split_subjects(subjects: np.ndarray, ratios: SplitRatios, seed: int = RANDOM_SEED) -> Dict[str, np.ndarray]:
    np.random.seed(seed)
    np.random.shuffle(subjects)

    total = len(subjects)
    n_train = int(total * ratios.train)
    n_val = int(total * ratios.val)

    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train : n_train + n_val]
    test_subjects = subjects[n_train + n_val :]

    logger.info(
        "Subject split -> train: %d | val: %d | test: %d",
        len(train_subjects),
        len(val_subjects),
        len(test_subjects),
    )

    return {
        "train": set(train_subjects),
        "val": set(val_subjects),
        "test": set(test_subjects),
    }


def assign_splits(df: pd.DataFrame, subject_splits: Dict[str, Set]) -> pd.Series:
    def _get_split(subject_id) -> str:
        for split_name, subject_set in subject_splits.items():
            if subject_id in subject_set:
                return split_name
        return "test"

    df_copy = df.copy()
    df_copy["split"] = df_copy["subject_id"].apply(_get_split)
    return df_copy


def compute_class_distribution(df: pd.DataFrame) -> Dict[str, Dict]:
    distribution = {}
    for split_name in SPLIT_NAMES:
        subset = df[df["split"] == split_name]
        if len(subset) == 0:
            continue

        label_counts = subset["label"].value_counts()
        total = len(subset)

        distribution[split_name] = {
            "total": total,
            "class_0": label_counts.get(0, 0),
            "class_1": label_counts.get(1, 0),
            "pct_0": (label_counts.get(0, 0) / total * 100) if total > 0 else 0.0,
            "pct_1": (label_counts.get(1, 0) / total * 100) if total > 0 else 0.0,
        }

    return distribution


def print_distribution_report(distribution: Dict[str, Dict]) -> None:
    logger.info("Distribution Report:")
    for split_name in SPLIT_NAMES:
        if split_name not in distribution:
            continue
        stats = distribution[split_name]
        logger.info(
            "[%s] Total: %d | Class-0: %d (%.1f%%) | Class-1: %d (%.1f%%)",
            split_name.upper(),
            stats["total"],
            stats["class_0"],
            stats["pct_0"],
            stats["class_1"],
            stats["pct_1"],
        )


def save_split_index(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
    logger.info("Split index saved to %s", output_path)


def perform_subject_split(
    index_path: str = INDEX_PATH,
    output_path: str = SPLIT_INDEX_PATH,
    ratios: SplitRatios = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    if ratios is None:
        ratios = SplitRatios(*DEFAULT_SPLIT_RATIOS)

    df = load_index(index_path)
    subjects = extract_subjects(df)
    subject_splits = split_subjects(subjects, ratios, seed)
    df = assign_splits(df, subject_splits)
    save_split_index(df, output_path)

    distribution = compute_class_distribution(df)
    print_distribution_report(distribution)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Performs subject-wise splitting of the compiled dataset.")
    parser.add_argument("--index-path", default=INDEX_PATH, help="Path to the dataset index")
    parser.add_argument("--output-path", default=SPLIT_INDEX_PATH, help="Output path for split index")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train fraction (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation fraction (default: 0.15)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed (default: 42)")
    args = parser.parse_args()

    ratios = SplitRatios(train=args.train_ratio, val=1.0 - args.train_ratio - args.val_ratio, test=args.val_ratio)
    perform_subject_split(
        index_path=args.index_path,
        output_path=args.output_path,
        ratios=ratios,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()