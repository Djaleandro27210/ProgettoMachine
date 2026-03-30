"""
=====================================================================================
Modulo: shap_fase2_early.py
Progetto: ML Emozioni - Fase 3 (Explainable AI su EARLY FUSION)

Descrizione:
Analisi SHAP multimodale definitiva. Segnali reali in NERO per massimo contrasto.
Legenda integrata e percorsi blindati.
=====================================================================================
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

# =========================================================
# 1. FIX DEI PERCORSI (PATH) 
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) 
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src", "Fase2", "early_fusion"))

try:
    from config_FASE2_early import BATCH_SIZE, MODEL_SAVE_PATH_FASE2, WINDOW_SIZE
    from dataset_FASE2_early import PopaneDatasetMultimodal
    from model_FASE2_early import MultimodalEarlyFusionCNN
    import shap
    print("✅ Moduli Early Fusion caricati con successo!")
except ImportError as e:
    print(f"❌ Errore critico: Impossibile trovare i file della Fase 2 Early. Errore: {e}")
    sys.exit(1)

def esegui_shap_fase2_early():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Avvio Analisi SHAP Early Fusion. Segnali in NERO. Dispositivo: {device}")

    # 1. CARICAMENTO MODELLO E DATI
    test_dataset = PopaneDatasetMultimodal(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    model = MultimodalEarlyFusionCNN().to(device)
    if not os.path.exists(MODEL_SAVE_PATH_FASE2):
        print(f"❌ Errore: Modello non trovato in {MODEL_SAVE_PATH_FASE2}")
        return
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH_FASE2, map_location=device, weights_only=True))
    model.eval()

    # 2. PREPARAZIONE DATI PER SHAP
    data_iterator = iter(test_loader)
    background_inputs, _ = next(data_iterator)
    background_inputs = background_inputs.to(device)

    test_inputs, test_labels = next(data_iterator)
    test_inputs = test_inputs[:10].to(device) 
    test_labels = test_labels[:10].cpu().numpy()

    # 3. CALCOLO VALORI SHAP
    print("🧠 Inizializzazione GradientExplainer...")
    explainer = shap.GradientExplainer(model, background_inputs)
    
    print("🔬 Calcolo SHAP values (Fase multimodale)...")
    shap_values = explainer.shap_values(test_inputs)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    test_inputs_np = test_inputs.cpu().numpy()
    shap_values_np = np.array(shap_values)

    # 4. GESTIONE CARTELLA PLOTS
    plots_dir = os.path.join(current_dir, 'plots_early_fusion')
    os.makedirs(plots_dir, exist_ok=True)

    # 5. GENERAZIONE GRAFICI
    sensor_names = ["AFFECT (Viso/Postura)", "ECG (Cuore)", "EDA (Sudore)"]
    
    for i in range(10):
        label_vera = int(test_labels[i])
        with torch.no_grad():
            output = model(test_inputs[i:i+1])
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0

        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f"Analisi SHAP EARLY FUSION - Paziente {i+1}\nVero: {label_vera} | Predetto: {pred} (Prob: {prob:.2f})", 
                     fontsize=16, fontweight='bold')

        legend_handles = []

        for channel in range(10):
            ax = axes[channel]
            signal = test_inputs_np[i, channel, :].flatten()
            shaps = shap_values_np[i, channel, :].flatten()
            
            # --- SEGNALE REALE IN NERO ---
            line_reale, = ax.plot(signal, color='black', alpha=0.6, linewidth=1.5, label='Segnale Reale')
            if channel == 0: legend_handles.append(line_reale)
            
            # --- BARRE SHAP ---
            colors = ['red' if float(val) > 0 else 'blue' for val in shaps]
            ax.bar(range(WINDOW_SIZE), shaps * 10, color=colors, width=1.0, alpha=0.7, edgecolor='none')
            
            ax.set_title(f"Sensore: {sensor_names[channel]}", fontsize=12, loc='left', fontweight='bold')
            ax.set_ylabel("Impatto SHAP")
            ax.grid(True, alpha=0.2, linestyle='--')

        # Proxy per la legenda colori
        red_proxy = mpatches.Rectangle((0,0),1,1, color='red', alpha=0.7, label='Spinge verso Positivo (1)')
        blue_proxy = mpatches.Rectangle((0,0),1,1, color='blue', alpha=0.7, label='Spinge verso Negativo (0)')
        legend_handles.extend([red_proxy, blue_proxy])
        
        fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=True, facecolor='white', bbox_to_anchor=(0.5, 0.01))

        axes[2].set_xlabel("Tempo (ms)")
        plt.tight_layout(rect=[0, 0.05, 1, 0.94])
        
        save_path = os.path.join(plots_dir, f'shap_early_paziente_{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ [SISTEMATO] Salvato: {save_path}")

    print(f"\n🎉 Analisi completata con successo! I grafici neri sono pronti in: {plots_dir}")

if __name__ == "__main__":
    esegui_shap_fase2_early()