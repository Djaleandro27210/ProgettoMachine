"""
create_index.py
Scans the raw dataset directory, extracts metadata, applies windowing, 
and generates a consolidated CSV index for lazy loading.
Incorporates undersampling to balance the positive/negative classes.
"""

import os
import glob
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any

from config import RAW_DIR, PROCESSED_DIR, WINDOW_SIZE, EMOTION_MAP, INDEX_PATH

# Constants specific to this dataset's structure
HEADER_LINES = 9


def _get_emotion_from_filename(filename: str) -> Optional[str]:
    """Extracts the emotion string from the filename formatted as ID_emotion.csv."""
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_', 1)
    
    if len(parts) < 2:
        return None
        
    return parts[1].lower()


def _extract_file_metadata(filepath: str) -> Tuple[Optional[str], int]:
    """Reads the file iteratively to extract the Subject ID and count data rows."""
    subject_id = None
    total_lines = 0
    
    # Read line-by-line to prevent memory saturation with large CSVs
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 1:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    subject_id = parts[1]
            total_lines += 1
            
    data_rows = max(0, total_lines - HEADER_LINES)
    return subject_id, data_rows


def create_dataset_index() -> None:
    print("[INFO] Scanning dataset directory for raw CSV files...")
    
    search_pattern = os.path.join(RAW_DIR, "**", "*.csv")
    all_csv_files = glob.glob(search_pattern, recursive=True)
    
    print(f"[INFO] Found {len(all_csv_files)} files. Initiating metadata extraction...")

    index_data: List[Dict[str, Any]] = []

    for filepath in all_csv_files:
        filename = os.path.basename(filepath)
        emotion_str = _get_emotion_from_filename(filename)
        
        if not emotion_str or emotion_str not in EMOTION_MAP:
            continue
            
        label = EMOTION_MAP[emotion_str]
        subject_id, data_rows = _extract_file_metadata(filepath)
        
        if data_rows < WINDOW_SIZE:
            continue
            
        num_windows = data_rows // WINDOW_SIZE
        
        for w in range(num_windows):
            # Target class undersampling: retain only 1 in 3 windows for the positive class
            if label == 1 and w % 3 != 0:
                continue 

            start_row = HEADER_LINES + (w * WINDOW_SIZE)
            end_row = start_row + WINDOW_SIZE
            
            index_data.append({
                'file_path': filepath,
                'subject_id': subject_id,
                'emotion': emotion_str,
                'label': label,
                'start_row': start_row,
                'end_row': end_row
            })

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_index = pd.DataFrame(index_data)
    
    output_path = INDEX_PATH if INDEX_PATH else os.path.join(PROCESSED_DIR, "dataset_index.csv")
    df_index.to_csv(output_path, index=False)
    
    print(f"[INFO] Indexing complete. Generated {len(df_index)} total training windows.")
    print(f"[INFO] Index successfully saved to: {output_path}")


if __name__ == "__main__":
    create_dataset_index()