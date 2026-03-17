"""
=====================================================================================
Orchestratore Progetto Emozioni
Avvia in sequenza tutta la pipeline: Split Dati -> Training -> Testing
=====================================================================================
"""
import subprocess
import sys
import time

def run_script(script_path):
    print(f"\n{'='*60}")
    print(f"🚀 AVVIO FASE 2: {script_path}")
    print(f"{'='*60}\n")
    
    # Esegue lo script come se lo lanciassi tu dal terminale
    result = subprocess.run([sys.executable, script_path])
    
    # Se lo script crasha (es. errore nan o file non trovato), ferma tutto!
    if result.returncode != 0:
        print(f"\n❌ ERRORE CRITICO: La fase '{script_path}' è fallita. Pipeline interrotta.")
        sys.exit(1)
    
    print(f"\n✅ FASE COMPLETATA: {script_path}")
    time.sleep(3) # Pausa di 3 secondi per far raffreddare la CPU e svuotare la RAM

if __name__ == "__main__":
    print("🔥 INIZIO PIPELINE COMPLETA - PROGETTO ML 🔥")
    print("Mettiti comodo, il PC lavorerà per un bel po'...\n")
    
    # --- FASE 1: PREPARAZIONE DATI ---
    # Scommenta le righe sotto se vuoi che ricrei l'indice da zero ogni volta
    # run_script("src/create_index.py")
    # run_script("src/split_data.py")
    
    # --- FASE 2: ADDESTRAMENTO ---
    run_script("src/Fase2/early_fusion/train_FASE2_early.py")
    
    # --- FASE 3: TEST FINALE ---
    run_script("src/Fase2/early_fusion/test_FASE2_early.py")
    # --- FASE 4: VALUTAZIONE COMPLETA E STAMPA RISULTATI ---
    # Sostituisci "src/Fase1/evaluate.py" con il percorso esatto dove hai salvato 
    # il codice di "evaluate.py" (quello con valuta_modello)
    run_script("src/Fase2/early_fusion/evaluate.py")
    print("\n🎉 TUTTA LA PIPELINE È TERMINATA CON SUCCESSO! IL MODELLO È PRONTO! 🎉")