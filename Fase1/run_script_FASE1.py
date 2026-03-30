"""
Phase 1 Pipeline Orchestrator - Training and Evaluation

This script handles the complete Phase 1 workflow: data preparation (optional),
model training, and evaluation. It can be run standalone or as part of the
global pipeline.
"""

import argparse
import logging
import subprocess
import sys
import time
from typing import List


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 70
COOLDOWN_SECONDS = 3


def print_header(message: str) -> None:
    """Print a formatted section header.
    
    Args:
        message: Header text to display.
    """
    print(f"\n{SEPARATOR}")
    print(f"{message}")
    print(f"{SEPARATOR}\n")


def run_script(script_path: str) -> bool:
    """Execute a Python script as a subprocess.
    
    Args:
        script_path: Path to the Python script to execute.
        
    Returns:
        True if script succeeded, False otherwise.
    """
    logger.info(f"[EXEC] Starting: {script_path}")
    
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode != 0:
        logger.error(f"[FATAL] Script failed: {script_path}")
        return False
    
    logger.info(f"[OK] Completed: {script_path}")
    logger.info(f"[INFO] Pausing {COOLDOWN_SECONDS}s for system to cool down...")
    time.sleep(COOLDOWN_SECONDS)
    
    return True


def run_phase1_pipeline(
    prepare_data: bool = False,
    train: bool = True,
    evaluate: bool = True,
) -> bool:
    """Execute Phase 1 training and evaluation pipeline.
    
    Args:
        prepare_data: If True, recreate index and split data from scratch.
        train: If True, run training script.
        evaluate: If True, run evaluation script.
        
    Returns:
        True if all enabled steps succeeded, False otherwise.
    """
    print_header("PHASE 1 PIPELINE - ECG-only Model")
    print("Grab a coffee, this will take a while...\n")
    
    steps = []
    
    if prepare_data:
        steps.extend([
            ("Data Preparation - Index", "src/create_index.py"),
            ("Data Preparation - Split", "src/split_data.py"),
        ])
    
    if train:
        steps.append(("Training", "src/Fase1/train_FASE1.py"))
    
    if evaluate:
        steps.append(("Evaluation", "src/Fase1/evaluate.py"))
    
    for step_name, script_path in steps:
        logger.info(f"Starting step: {step_name}")
        success = run_script(script_path)
        if not success:
            logger.error(f"Pipeline failed at step: {step_name}")
            return False
    
    return True


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Phase 1 Pipeline - ECG-only model training and evaluation"
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Recreate index and split from scratch (optional)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run evaluation",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    args = parser.parse_args()
    
    success = run_phase1_pipeline(
        prepare_data=args.prepare_data,
        train=not args.skip_train,
        evaluate=not args.skip_eval,
    )
    
    if success:
        print_header("PHASE 1 PIPELINE COMPLETED SUCCESSFULLY")
        print("Model is ready for deployment or further analysis.\n")
        sys.exit(0)
    else:
        print_header("PHASE 1 PIPELINE FAILED")
        print("Check logs above for error details.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()