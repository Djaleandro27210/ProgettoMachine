"""
=====================================================================================
Emotion Project Orchestrator
Sequentially starts the entire pipeline: Data Split -> Training -> Testing
=====================================================================================
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def run_script(script_path: Path, delay: float = 2.0, verbose: bool = True) -> bool:
    """Executes a Python script and returns the success status."""
    if verbose:
        print("\n" + "=" * 60)
        print(f"RUNNING SCRIPT: {script_path}")
        print("" + "=" * 60 + "\n")

    if not script_path.exists():
        print(f"ERROR: File not found: {script_path}")
        return False

    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)

    if verbose and result.stdout:
        print("OUTPUT:\n", result.stdout)

    if result.returncode != 0:
        print(f"\nERROR: Script '{script_path}' terminated with code {result.returncode}.")
        if verbose and result.stderr:
            print("STDERR:\n", result.stderr)
        return False

    if verbose:
        print(f"\nSCRIPT COMPLETED: {script_path}")

    if delay > 0:
        time.sleep(delay)

    return True


def build_pipeline(default_scripts: List[Path], selected_steps: Optional[List[str]] = None) -> List[Path]:
    """Returns the script pipeline configured according to the selected steps."""
    if selected_steps is None or "all" in selected_steps:
        return default_scripts

    mapping = {script.name.split("_")[0]: script for script in default_scripts}
    selected = []
    for step in selected_steps:
        if step in mapping:
            selected.append(mapping[step])
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runs the Phase 2 Early Fusion pipeline for ML Emotions")
    parser.add_argument("--steps", nargs="*", default=["train", "evaluate"],
                        choices=["train", "evaluate", "all"],
                        help="Steps to execute: train, evaluate or all")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to pause between steps")
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose output")
    return parser.parse_args()


def main() -> int:
    print("STARTING PHASE 2 EARLY FUSION PIPELINE")

    script_dir = Path(__file__).resolve().parent
    pipeline_scripts = [
        script_dir / "train_FASE2_early.py",
        script_dir / "evaluate.py",
    ]

    args = parse_args()
    verbose = not args.no_verbose
    selected = args.steps if "all" not in args.steps else ["train", "evaluate"]

    if not selected:
        print("No steps selected. Using train + evaluate by default")
        selected = ["train", "evaluate"]

    run_list = build_pipeline(pipeline_scripts, selected)

    for script_path in run_list:
        if not script_path.exists():
            print(f"File not found: {script_path}")
            return 1

        success = run_script(script_path, delay=args.sleep, verbose=verbose)
        if not success:
            print(f"Pipeline interrupted due to error in: {script_path}")
            return 1

    print("\nPIPELINE SUCCESSFULLY COMPLETED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

if __name__ == "__main__":
    main()
