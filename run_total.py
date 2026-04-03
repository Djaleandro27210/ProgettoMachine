"""
master_orchestrator.py
Master sequential orchestrator for the entire ML project.
Executes Phase 1, Phase 2 (Early & Late Fusion), and Phase 3 (SHAP XAI) sequentially.
"""

import subprocess
import sys
import time

COOLDOWN_SECONDS = 10


def run_script(script_path: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"[INFO] Executing: {script_path}")
    print(f"{'-' * 60}\n")
    
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode != 0:
        print(f"\n[CRITICAL ERROR] Execution failed for '{script_path}'. Pipeline halted.")
        sys.exit(1)
    
    print(f"\n[INFO] Successfully completed: {script_path}")
    print(f"[INFO] Cooling down system for {COOLDOWN_SECONDS} seconds...\n")
    time.sleep(COOLDOWN_SECONDS)


def run_master_pipeline() -> None:
    print("[INFO] Initializing Master ML Pipeline...")
    print("[INFO] This process will take significant time. Please stand by.\n")
    
    # --- Phase 1: Unimodal (ECG) ---
    print(">>> STARTING PHASE 1: UNIMODAL <<<")
    run_script("src/Fase1/train_FASE1.py")
    run_script("src/Fase1/evaluate.py")
    
    # --- Phase 2: Early Fusion ---
    print("\n>>> STARTING PHASE 2: EARLY FUSION <<<")
    run_script("src/Fase2/early_fusion/train_FASE2_early.py")
    run_script("src/Fase2/early_fusion/evaluate.py")
    
    # --- Phase 2: Late Fusion ---
    print("\n>>> STARTING PHASE 2: LATE FUSION <<<")
    run_script("src/Fase2/late_fusion/train_FASE2_late.py")
    run_script("src/Fase2/late_fusion/evaluate_late.py")

    # --- Phase 3: Explainable AI (SHAP) ---
    print("\n>>> STARTING PHASE 3: SHAP EXPLAINABILITY <<<")
    run_script("src/Fase3/shap_fase1.py")
    run_script("src/Fase3/shap_fase2_early.py")
    run_script("src/Fase3/shap_fase2_late.py")

    print("\n[SUCCESS] Master pipeline completed. All models trained, evaluated, and explained.")
    print("[INFO] Please review the generated output text files and SHAP plots.")


if __name__ == "__main__":
    run_master_pipeline()