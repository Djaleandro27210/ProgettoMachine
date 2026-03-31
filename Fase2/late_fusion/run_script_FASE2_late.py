"""
=====================================================================================
Emotions Project Orchestrator
Sequentially launches the entire pipeline: Data Split -> Training -> Testing
=====================================================================================
"""
import subprocess
import sys
import time

def run_script(script_path):
    print(f"\n{'='*60}")
    print(f"STARTING PHASE 2: {script_path}")
    print(f"{'='*60}\n")
    
    # Executes the script as if you were launching it from the terminal
    result = subprocess.run([sys.executable, script_path])
    
    # If the script crashes (e.g., nan error or file not found), stop everything!
    if result.returncode != 0:
        print(f"\nCRITICAL ERROR: Phase '{script_path}' failed. Pipeline interrupted.")
        sys.exit(1)
    
    print(f"\nPHASE COMPLETED: {script_path}")
    time.sleep(3) # Pause of 3 seconds to cool down the CPU and clear RAM

if __name__ == "__main__":
    print("STARTING COMPLETE PIPELINE - ML PROJECT")
    print("Get comfortable, the PC will work for a while...\n")
    
    # --- PHASE 1: DATA PREPARATION ---
    # Uncomment the lines below if you want to recreate the index from scratch every time
    # run_script("src/create_index.py")
    # run_script("src/split_data.py")
    
    # --- PHASE 2: TRAINING ---
    run_script("src/Fase2/late_fusion/train_FASE2_late.py")
    

    # --- PHASE 4: COMPLETE EVALUATION AND PRINT RESULTS ---
    # Replace "src/Fase1/evaluate.py" with the exact path where you saved 
    # the code of "evaluate.py" (the one with evaluate_model)
    run_script("src/Fase2/late_fusion/evaluate.py")
    print("\nTHE ENTIRE PIPELINE HAS FINISHED SUCCESSFULLY! THE MODEL IS READY!")