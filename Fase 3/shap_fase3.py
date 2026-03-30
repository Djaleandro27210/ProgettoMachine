"""
=====================================================================================
Modulo: shap_fase2_late.py
Progetto: ML Emozioni - Fase 3 (Explainable AI su LATE FUSION)

Descrizione:
Interroga i "3 Giudici" della Late Fusion. Spiega la decisione individuale di 
ogni modello (Affect, ECG, EDA) prima del voto a maggioranza.
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
sys.path.append(os.path.join(project_root, "src", "Fase2", "late_fusion"))

try:
    from config_FASE2_late import (BATCH_SIZE, WINDOW_SIZE,
                                   MODEL_SAVE_PATH_LATE_AFFECT, 
                                   MODEL_SAVE_PATH_LATE_ECG, 
                                   MODEL_SAVE_PATH_LATE_EDA)
    from dataset_FASE2_late import PopaneDatasetLateFusion
    from model_FASE2_late import UnimodalCNN
    import shap
    print("✅ Moduli Late Fusion caricati!")
except ImportError as e:
    print(f"❌ Errore critico: {e}")
    sys.exit(1)

def esegui_shap_fase2_late():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Avvio SHAP Late Fusion. Analisi dei 3 Giudici. Dispositivo: {device}")

    # 1. CARICAMENTO DEI 3 MODELLI (I GIUDICI)
    test_dataset = PopaneDatasetLateFusion(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    def load_model(path):
        m = UnimodalCNN().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        return m

    model_aff = load_model(MODEL_SAVE_PATH_LATE_AFFECT)
    model_ecg = load_model(MODEL_SAVE_PATH_LATE_ECG)
    model_eda = load_model(MODEL_SAVE_PATH_LATE_EDA)

    # 2. PREPARAZIONE DATI
    data_iterator = iter(test_loader)
    back_aff, back_ecg, back_eda, _ = next(data_iterator)
    
    t_aff, t_ecg, t_eda, t_labels = next(data_iterator)
    t_aff, t_ecg, t_eda = t_aff[:3].to(device), t_ecg[:3].to(device), t_eda[:3].to(device)
    t_labels = t_labels[:3].numpy()

    # 3. CREAZIONE DI 3 EXPLAINER (Uno per giudice)
    print("🧠 Inizializzazione dei 3 Explainers...")
    exp_aff = shap.GradientExplainer(model_aff, back_aff.to(device))
    exp_ecg = shap.GradientExplainer(model_ecg, back_ecg.to(device))
    exp_eda = shap.GradientExplainer(model_eda, back_eda.to(device))

    # 4. CALCOLO VALORI SHAP
    print("🔬 Calcolo dei voti e delle spiegazioni...")
    s_aff = np.array(exp_aff.shap_values(t_aff))
    s_ecg = np.array(exp_ecg.shap_values(t_ecg))
    s_eda = np.array(exp_eda.shap_values(t_eda))

    # 5. GENERAZIONE GRAFICI
    plots_dir = os.path.join(current_dir, 'plots_late_fusion')
    os.makedirs(plots_dir, exist_ok=True)

    for i in range(3):
        # Calcolo voti singoli
        with torch.no_grad():
            v_aff = 1 if torch.sigmoid(model_aff(t_aff[i:i+1])).item() > 0.5 else 0
            v_ecg = 1 if torch.sigmoid(model_ecg(t_ecg[i:i+1])).item() > 0.5 else 0
            v_eda = 1 if torch.sigmoid(model_eda(t_eda[i:i+1])).item() > 0.5 else 0
        
        voto_finale = 1 if (v_aff + v_ecg + v_eda) >= 2 else 0
        label_vera = int(t_labels[i])

        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f"Analisi SHAP LATE FUSION - Paziente {i+1}\nVero: {label_vera} | Verdetto Finale: {voto_finale}", 
                     fontsize=16, fontweight='bold')

        # Dati per il loop
        data_list = [
            (t_aff[i].cpu().numpy().flatten(), np.squeeze(s_aff[i]).flatten(), v_aff, "AFFECT"),
            (t_ecg[i].cpu().numpy().flatten(), np.squeeze(s_ecg[i]).flatten(), v_ecg, "ECG"),
            (t_eda[i].cpu().numpy().flatten(), np.squeeze(s_eda[i]).flatten(), v_eda, "EDA")
        ]

        legend_handles = []

        for ch, (signal, shaps, voto, name) in enumerate(data_list):
            ax = axes[ch]
            # Segnale Nero
            line, = ax.plot(signal, color='black', alpha=0.6, linewidth=1.5, label='Segnale Reale')
            if ch == 0: legend_handles.append(line)

            # Barre SHAP (moltiplicate per 10 per visibilità come nell'early)
            colors = ['red' if float(val) > 0 else 'blue' for val in shaps]
            ax.bar(range(WINDOW_SIZE), shaps * 10, color=colors, width=1.0, alpha=0.7, edgecolor='none')
            
            ax.set_title(f"Giudice {name} | Voto Individuale: {voto}", fontsize=12, loc='left', fontweight='bold')
            ax.set_ylabel("Impatto SHAP")
            ax.grid(True, alpha=0.2, linestyle='--')

        # Legenda
        red_p = mpatches.Rectangle((0,0),1,1, color='red', alpha=0.7, label='Spinge verso Positivo (1)')
        blue_p = mpatches.Rectangle((0,0),1,1, color='blue', alpha=0.7, label='Spinge verso Negativo (0)')
        legend_handles.extend([red_p, blue_p])
        fig.legend(handles=legend_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))

        plt.tight_layout(rect=[0, 0.05, 1, 0.94])
        save_path = os.path.join(plots_dir, f'shap_late_paziente_{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Analisi Giudici salvata: {save_path}")

if __name__ == "__main__":
    esegui_shap_fase2_late()

"""
=====================================================================================
📖 GUIDA ALL'INTERPRETAZIONE (LATE FUSION - IL TRIBUNALE)
=====================================================================================

Qui non vediamo un'unica mente, ma tre pareri distinti.

1. IL CONFLITTO DEI GIUDICI:
   - È possibile che il Giudice ECG abbia barre tutte BLU (voto 0) e il Giudice EDA 
     abbia barre tutte ROSSE (voto 1). In questo caso, lo SHAP ti mostra 
     esattamente cosa ha convinto uno e cosa ha ingannato l'altro.

2. LA FORZA DEL VOTO:
   - Se un giudice vota 1 (Positivo) ma le sue barre SHAP sono bassissime, 
     significa che è un voto "debole", quasi incerto. 
   - Se le barre sono altissime, il giudice è convinto della sua posizione.

3. CONFRONTO CON EARLY FUSION:
   - Nota se i punti di importanza (le barre colorate) sono negli stessi posti della 
     Early Fusion. Spesso nella Late Fusion i modelli sono più "semplici" e 
     guardano meno dettagli rispetto alla Early Fusion, spiegando perché 
     l'accuratezza generale è leggermente più bassa.
=====================================================================================
"""