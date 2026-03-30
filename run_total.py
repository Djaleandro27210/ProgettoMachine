"""
Pipeline Orchestrator - Sequential Safe Execution
Launch and monitor all training, evaluation, and analysis phases.
"""
import argparse
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 70
COOLDOWN_SECONDS = 10


@dataclass(frozen=True)
class PipelineStage:
    name: str
    scripts: List[str]


def print_header(message: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"{message}")
    print(f"{SEPARATOR}\n")


def run_script(script_path: str, verbose: bool = True) -> bool:
    if verbose:
        print(f"[EXEC] Avvio: {script_path}")
    
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode != 0:
        print(f"[ERROR] Script fallito: {script_path}")
        return False
    
    print(f"[OK] Completato: {script_path}")
    if verbose:
        print(f"[INFO] Pausa {COOLDOWN_SECONDS}s per raffreddare sistema...\n")
        time.sleep(COOLDOWN_SECONDS)
    
    return True


def run_stage(stage: PipelineStage, stop_on_error: bool = True) -> bool:
    print_header(f"FASE: {stage.name}")
    
    for script_path in stage.scripts:
        success = run_script(script_path)
        if not success and stop_on_error:
            print(f"[FATAL] Pipeline interrotta a causa di errore in {script_path}")
            return False
    
    return True


def execute_pipeline(stages: List[PipelineStage], stop_on_error: bool = True) -> bool:
    print_header("INIZIO PIPELINE COMPLETA")
    print("Mettiti comodo, ci vediamo tra un paio di giorni...\n")
    
    for stage in stages:
        success = run_stage(stage, stop_on_error=stop_on_error)
        if not success:
            return False
    
    return True


def get_default_pipeline() -> List[PipelineStage]:
    return [
        PipelineStage(
            name="FASE 1 - Unimodale",
            scripts=[
                "src/Fase1/train_FASE1.py",
                "src/Fase1/evaluate.py",
            ],
        ),
        PipelineStage(
            name="FASE 2 - Early Fusion",
            scripts=[
                "src/Fase2/early_fusion/train_FASE2_early.py",
                "src/Fase2/early_fusion/evaluate.py",
            ],
        ),
        PipelineStage(
            name="FASE 2 - Late Fusion",
            scripts=[
                "src/Fase2/late_fusion/train_FASE2_late.py",
                "src/Fase2/late_fusion/evaluate_late.py",
            ],
        ),
        PipelineStage(
            name="FASE 3 - Analisi SHAP",
            scripts=[
                "src/Fase3/shap_fase1.py",
                "src/Fase3/shap_fase2_early.py",
                "src/Fase3/shap_fase2_late.py",
            ],
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestratore di pipeline - esecuzione sequenziale con monitoring"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["fase1", "fase2-early", "fase2-late", "fase3", "all"],
        default="all",
        help="Quale fase eseguire (default: all)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continua anche se uno script fallisce",
    )
    args = parser.parse_args()

    all_stages = get_default_pipeline()
    
    if args.stage != "all":
        stage_map = {
            "fase1": 0,
            "fase2-early": 1,
            "fase2-late": 2,
            "fase3": 3,
        }
        stages = [all_stages[stage_map[args.stage]]]
    else:
        stages = all_stages

    success = execute_pipeline(stages, stop_on_error=not args.continue_on_error)
    
    if success:
        print_header("MISSIONE COMPIUTA - TUTTI GLI STEP COMPLETATI")
        print("Controlla i file output_fase1.txt, output_fase2_early.txt e output_fase2_late.txt\n")
        sys.exit(0)
    else:
        print_header("PIPELINE FALLITA - ERRORE DURANTE L'ESECUZIONE")
        sys.exit(1)


if __name__ == "__main__":
    main()