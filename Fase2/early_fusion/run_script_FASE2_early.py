"""
=====================================================================================
Orchestratore Progetto Emozioni
Avvia in sequenza tutta la pipeline: Split Dati -> Training -> Testing
=====================================================================================
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def run_script(script_path: Path, delay: float = 2.0, verbose: bool = True) -> bool:
    """Esegue uno script Python e restituisce lo stato di riuscita."""
    if verbose:
        print("\n" + "=" * 60)
        print(f"ESECUZIONE SCRIPT: {script_path}")
        print("" + "=" * 60 + "\n")

    if not script_path.exists():
        print(f"ERROR: file non trovato: {script_path}")
        return False

    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)

    if verbose and result.stdout:
        print("OUTPUT:\n", result.stdout)

    if result.returncode != 0:
        print(f"\nERRORE: Lo script '{script_path}' è terminato con codice {result.returncode}.")
        if verbose and result.stderr:
            print("STDERR:\n", result.stderr)
        return False

    if verbose:
        print(f"\nSCRIPT COMPLETATO: {script_path}")

    if delay > 0:
        time.sleep(delay)

    return True


def build_pipeline(default_scripts: List[Path], selected_steps: Optional[List[str]] = None) -> List[Path]:
    """Restituisce la pipeline di script configurata secondo i passi selezionati."""
    if selected_steps is None or "all" in selected_steps:
        return default_scripts

    mapping = {script.name.split("_")[0]: script for script in default_scripts}
    selected = []
    for step in selected_steps:
        if step in mapping:
            selected.append(mapping[step])
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Esegue la pipeline Fase2 Early Fusion per ML Emozioni")
    parser.add_argument("--steps", nargs="*", default=["train", "evaluate"],
                        choices=["train", "evaluate", "all"],
                        help="I passi da eseguire: train, evaluate o all")
    parser.add_argument("--sleep", type=float, default=2.0, help="Secondi di pausa tra i passi")
    parser.add_argument("--no-verbose", action="store_true", help="Disabilita output dettagliato")
    return parser.parse_args()


def main() -> int:
    print("AVVIO PIPELINE FASE2 EARLY FUSION")

    script_dir = Path(__file__).resolve().parent
    pipeline_scripts = [
        script_dir / "train_FASE2_early.py",
        script_dir / "evaluate.py",
    ]

    args = parse_args()
    verbose = not args.no_verbose
    selected = args.steps if "all" not in args.steps else ["train", "evaluate"]

    if not selected:
        print("Nessun passo selezionato. Uso train + evaluate per default")
        selected = ["train", "evaluate"]

    run_list = build_pipeline(pipeline_scripts, selected)

    for script_path in run_list:
        if not script_path.exists():
            print(f"File non trovato: {script_path}")
            return 1

        success = run_script(script_path, delay=args.sleep, verbose=verbose)
        if not success:
            print(f"Pipeline interrotta a causa di errore in: {script_path}")
            return 1

    print("\nPIPELINE COMPLETA ESEGUITA CON SUCCESSO")
    return 0


if __name__ == "__main__":
    sys.exit(main())

if __name__ == "__main__":
    main()
