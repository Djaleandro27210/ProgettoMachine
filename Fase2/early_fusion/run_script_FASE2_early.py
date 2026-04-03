"""
orchestrator_FASE2_early.py
Sequential orchestrator for Phase 2 (Early Fusion). 
Executes data preparation, training, and evaluation scripts safely.
"""

import subprocess
import sys
import time

COOLDOWN_SECONDS = 3


def run_script(script_path: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"[INFO] Executing: {script_path}")
    print(f"{'-' * 60}\n")
    
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode != 0:
        print(f"\n[ERROR] Execution failed for '{script_path}'. Halting pipeline.")
        sys.exit(1)
    
    print(f"\n[INFO] Successfully completed: {script_path}")
    time.sleep(COOLDOWN_SECONDS)


def run_pipeline() -> None:
    print("[INFO] Initializing Phase 2 Pipeline (Early Fusion)...\n")
    
    # --- Step 1: Data Preparation ---
    # Uncomment to recreate index and splits from scratch
    # run_script("src/create_index.py")
    # run_script("src/split_data.py")
    
    # --- Step 2: Training ---
    run_script("src/Fase2/early_fusion/train_FASE2_early.py")
    
    # --- Step 3: Evaluation ---
    run_script("src/Fase2/early_fusion/evaluate.py")
    
    print("\n[INFO] Phase 2 Pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()