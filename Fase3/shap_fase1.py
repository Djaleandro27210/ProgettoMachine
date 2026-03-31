"""
=====================================================================================
Modulo: shap_fase1.py
Progetto: ML Emozioni - Fase 3 (Explainable AI su ECG)

Descrizione:
Script definitivo per l'analisi SHAP. Spiega come la CNN interpreta l'ECG.
=====================================================================================
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# =========================================================
# 1. FIX DEI PERCORSI (PATH) - PER TROVARE LE ALTRE FASI
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) # Root del progetto
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src", "Fase1")) # Percorso alla Fase 1

try:
    from config_FASE1 import BATCH_SIZE, MODEL_SAVE_PATH, WINDOW_SIZE
    from dataset_FASE1 import PopaneDataset
    from model_FASE1 import Emotion1DCNN
    import shap
    print("✅ Moduli caricati con successo!")
except ImportError as e:
    print(f"❌ Errore critico: Impossibile trovare i file della Fase 1. Errore: {e}")
    sys.exit(1)

def esegui_shap_fase1():
    # Setup Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Avvio Analisi SHAP. Dispositivo: {device}")

    # 1. CARICAMENTO MODELLO E DATI
    test_dataset = PopaneDataset(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    model = Emotion1DCNN().to(device)
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"❌ Errore: Modello non trovato in {MODEL_SAVE_PATH}")
        return
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()
    print("✅ Modello Fase 1 caricato!") 
    

    # 2. PREPARAZIONE DATI PER SHAP
    data_iterator = iter(test_loader)
    background_inputs, _ = next(data_iterator)
    background_inputs = background_inputs.to(device)

    test_inputs, test_labels = next(data_iterator)
    test_inputs = test_inputs[:10].to(device) # Analizziamo i primi 10 pazienti
    test_labels = test_labels[:10].cpu().numpy()

    # 3. CALCOLO VALORI SHAP
    print("🧠 Inizializzazione GradientExplainer...")
    explainer = shap.GradientExplainer(model, background_inputs)
    
    print("🔬 Calcolo importanza feature (SHAP values)...")
    shap_values = explainer.shap_values(test_inputs)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Conversione per plotting
    test_inputs_np = test_inputs.cpu().numpy()
    shap_values_np = np.array(shap_values)

    # 4. GESTIONE CARTELLA PLOTS (Percorso Assoluto)
    plots_dir = os.path.join(current_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    print(f"📁 Cartella destinazione: {plots_dir}")

    # 5. GENERAZIONE GRAFICI
    print("🎨 Generazione grafici in corso...")
    
    for i in range(10):
        # Estrazione e pulizia dati (Fix per il TypeError)
        signal = test_inputs_np[i, 0, :].flatten()
        shaps = np.squeeze(shap_values_np[i, 0, :]).flatten()
        
        label_vera = int(test_labels[i])
        
        with torch.no_grad():
            output = model(test_inputs[i:i+1])
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0

        # Creazione Figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Analisi SHAP Paziente {i+1}\nVero: {label_vera} | Predetto: {pred} (Prob: {prob:.2f})", fontsize=14, fontweight='bold')

        # Subplot 1: Segnale ECG
        ax1.plot(signal, color='black', linewidth=1.2, label='Segnale ECG')
        ax1.set_title("Forma d'onda ECG (Normalizzata)")
        ax1.set_ylabel("Ampiezza")
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Valori SHAP
        # Definiamo i colori: Rosso spinge verso 1 (Positivo), Blu spinge verso 0 (Negativo)
        colors = ['red' if float(val) > 0 else 'blue' for val in shaps]
        ax2.bar(range(len(shaps)), shaps, color=colors, width=1.0, edgecolor='none')
        ax2.set_title("Contributo SHAP (Rosso: verso Positivo | Blu: verso Negativo)")
        ax2.set_xlabel("Tempo (ms)")
        ax2.set_ylabel("Impatto SHAP")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Salvataggio e verifica
        save_path = os.path.join(plots_dir, f'shap_fase1_paziente_{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        if os.path.exists(save_path):
            print(f"✅ [OK] Salvato: {save_path}")
        else:
            print(f"❌ [ERRORE] Fallito salvataggio in: {save_path}")

    print(f"\n🎉 ANALISI COMPLETATA! Trovi i file qui: {plots_dir}")

if __name__ == "__main__":
    esegui_shap_fase1()
    """
=====================================================================================
📖 GUIDA ALL'INTERPRETAZIONE DEI GRAFICI SHAP (XAI - EXPLAINABLE AI)
=====================================================================================

Questi grafici permettono di "aprire la scatola nera" della 1D-CNN, mostrando quali 
punti del segnale ECG hanno influenzato la decisione del modello.

1. IL SIGNIFICATO DEI COLORI:
   - ROSSO (SHAP > 0): Il modello ha interpretato questi millisecondi come indicatori
     di un'emozione POSITIVA (Classe 1). Più la barra è alta, più forte è l'influenza.
   - BLU (SHAP < 0): Il modello ha interpretato questi millisecondi come indicatori
     di un'emozione NEGATIVA (Classe 0). Più la barra scende, più "convinto" è il modello.

2. COSA OSSERVARE NELL'ONDA ECG:
   - COMPLESSO QRS (Il picco alto): È la parte più importante. Se è rosso, la rete 
     associa la forza/velocità della contrazione ventricolare al benessere. Se è blu,
     rileva anomalie nel picco che associa allo stress o al disagio.
   - INTERVALLO T (La gobba dopo il picco): Rappresenta il rilassamento del cuore. 
     Contributi significativi qui indicano che il modello sta analizzando la 
     'ripolarizzazione', un processo molto sensibile al sistema nervoso autonomo.
   - LINEA ISOELETTRICA (Le parti piatte): Rappresentano il ritmo. Se il modello 
     si colora qui, sta analizzando la HRV (Heart Rate Variability).

3. LOGICA MATEMATICA:
   Se la somma dei 'rossi' supera quella dei 'blu', il modello 
   predice un'emozione positiva.

4. ANALISI DEGLI ERRORI (Falsi Positivi/Negativi):
   Nei grafici dove il modello sbaglia (es. Vero: 0, Predetto: 1), SHAP rivelerà 
   quali artefatti o morfologie del segnale hanno 'ingannato' la rete, fornendo 
   una prova fondamentale per discutere i limiti del sistema unimodale.
=====================================================================================
"""