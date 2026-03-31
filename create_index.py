"""
Module: create_index.py
Project: ML Emotions - Dataset Indexing

Description:
This module creates an index for the dataset from raw CSV files. It scans the raw data directory,
parses emotion labels from filenames, and generates windowed entries for machine learning processing.
Supports undersampling of positive labels to balance the dataset.
"""

import argparse
import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

from config import EMOTION_MAP, INDEX_PATH, PROCESSED_DIR, RAW_DIR, WINDOW_SIZE


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexEntry:
    file_path: str
    subject_id: Optional[str]
    emotion: str
    label: int
    start_row: int
    end_row: int


def parse_emotion_from_filename(filepath: str, emotion_map: Dict[str, int]) -> Optional[Tuple[str, int]]:
    name_without_ext = Path(filepath).stem
    parts = name_without_ext.split("_", 1)
    if len(parts) != 2:
        logger.debug("Skipped file without emotion pattern: %s", filepath)
        return None

    emotion = parts[1].strip().lower()
    if emotion not in emotion_map:
        logger.debug("Skipped unmapped emotion: %s -> %s", filepath, emotion)
        return None

    return emotion, emotion_map[emotion]


def scan_file_metadata(filepath: str, header_lines: int = 9) -> Tuple[Optional[str], int]:
    subject_id = None
    total_lines = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 1:
                parts = [item.strip() for item in line.strip().split(",")]
                if len(parts) > 1:
                    subject_id = parts[1]
            total_lines += 1

    data_rows = max(0, total_lines - header_lines)
    return subject_id, data_rows


def generate_window_bounds(data_rows: int, window_size: int, start_offset: int = 9) -> Iterator[Tuple[int, int]]:
    for i in range(data_rows // window_size):
        start_row = start_offset + i * window_size
        end_row = start_row + window_size
        yield start_row, end_row


def should_keep_window(label: int, window_idx: int, undersample_positive: bool = True) -> bool:
    if not undersample_positive:
        return True
    if label == 1:
        return window_idx % 3 == 0
    return True


def build_index(undersample_positive: bool = True) -> pd.DataFrame:
    logger.info("Starting dataset scan in %s", RAW_DIR)

    search_pattern = os.path.join(RAW_DIR, "**", "*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    logger.info("Found %d CSV files", len(csv_files))

    entries: List[IndexEntry] = []

    for filepath in csv_files:
        parsed = parse_emotion_from_filename(filepath, EMOTION_MAP)
        if parsed is None:
            continue

        emotion, label = parsed
        subject_id, data_rows = scan_file_metadata(filepath)

        if data_rows < WINDOW_SIZE:
            logger.debug("[SKIP] %s with %d data rows < %d", filepath, data_rows, WINDOW_SIZE)
            continue

        for window_idx, (start_row, end_row) in enumerate(generate_window_bounds(data_rows, WINDOW_SIZE)):
            if not should_keep_window(label, window_idx, undersample_positive):
                continue

            entries.append(IndexEntry(
                file_path=filepath,
                subject_id=subject_id,
                emotion=emotion,
                label=label,
                start_row=start_row,
                end_row=end_row,
            ))

    df_index = pd.DataFrame([entry.__dict__ for entry in entries])
    logger.info("Finished – records created: %d", len(df_index))
    return df_index


def save_index(df_index: pd.DataFrame, output_dir: str = PROCESSED_DIR, index_filename: str = INDEX_PATH) -> str:
    os.makedirs(output_dir, exist_ok=True)
    df_index.to_csv(index_filename, index=False)
    logger.info("Index saved in %s", index_filename)
    return index_filename


def main() -> None:
    parser = argparse.ArgumentParser(description='Create the dataset index from raw CSV files.')
    parser.add_argument('--no-undersample-positive', action='store_true', help='Disable undersampling of label 1')
    args = parser.parse_args()

    df_index = build_index(undersample_positive=not args.no_undersample_positive)
    save_index(df_index)


if __name__ == "__main__":
    main()