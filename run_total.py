"""
=====================================================================================
Orchestratore SUPREMO - Esecuzione Sequenziale Sicura
Lancialo, esci di casa per 2 giorni, e torna a goderti i risultati.
=====================================================================================
"""
import subprocess
import sys
import time

def run_script(script_path):
    print(f"\n{'='*60}")
    print(f"🚀 AVVIO FASE: {script_path}")
    print(f"{'='*60}\n")
    
    # Esegue lo script
    result = subprocess.run([sys.executable, script_path])
    
    # Se crasha, ferma tutto e avvisa
    if result.returncode != 0:
        print(f"\n❌ ERRORE CRITICO in '{script_path}'. Processo interrotto.")
        sys.exit(1)
    
    print(f"\n✅ COMPLETATO: {script_path}")
    print("⏳ Pausa di 10 secondi per far raffreddare CPU e svuotare la RAM...\n")
    time.sleep(10) # 10 secondi di respiro per il PC

if __name__ == "__main__":
    print("🔥 INIZIO PIPELINE COMPLETA - MACCHINA AL LAVORO 🔥")
    print("Mettiti comodo, ci vediamo tra un paio di giorni...\n")
    
    # --- FASE 1 (Solo ECG) ---
    print(">>> PARTENZA FASE 1: UNIMODALE <<<")
    run_script("src/Fase1/train_FASE1.py")
    run_script("src/Fase1/evaluate.py")
    
    # --- FASE 2 EARLY FUSION (Il pastone) ---
    print("\n>>> PARTENZA FASE 2: EARLY FUSION <<<")
    run_script("src/Fase2/early_fusion/train_FASE2_early.py")
    run_script("src/Fase2/early_fusion/evaluate.py")
    
    # --- FASE 2 LATE FUSION (I 3 Giudici) ---
    print("\n>>> PARTENZA FASE 2: LATE FUSION <<<")
    run_script("src/Fase2/late_fusion/train_FASE2_late.py")
    run_script("src/Fase2/late_fusion/evaluate_late.py") # <-- Assicurati che il file del tribunale si chiami così!

    print("\n🎉 MISSIONE COMPIUTA! TUTTI I MODELLI SONO STATI ADDESTRATI E VALUTATI! 🎉")
    print("Controlla i file output_fase1.txt, output_fase2_early.txt e output_fase2_late.txt")

    run_script("src/Fase3/shap_fase1.py")
    run_script("src/Fase3/shap_fase2_early.py")
    run_script("src/Fase3/shap_fase2_late.py")