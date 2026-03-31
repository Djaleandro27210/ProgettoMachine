"""
=====================================================================================
Module: shap_fase2_late.py
Project: ML Emotions - Phase 3 (Explainable AI on LATE FUSION)

Description:
Interrogates the "3 Judges" of Late Fusion. Explains the individual decision of
each model (Affect, ECG, EDA) before the majority vote.
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
# 1. FIX PATHS
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
    print("Modules loaded successfully!")
except ImportError as e:
    print(f"Critical error: {e}")
    sys.exit(1)

def esegui_shap_fase2_late():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting SHAP Late Fusion. Analysis of the 3 Judges. Device: {device}")

    # 1. LOADING THE 3 MODELS (THE JUDGES)
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

    # 2. DATA PREPARATION
    data_iterator = iter(test_loader)
    back_aff, back_ecg, back_eda, _ = next(data_iterator)
    
    t_aff, t_ecg, t_eda, t_labels = next(data_iterator)
    t_aff, t_ecg, t_eda = t_aff[:10].to(device), t_ecg[:10].to(device), t_eda[:10].to(device)
    t_labels = t_labels[:10].numpy()

    # 3. CREATING 3 EXPLAINERS (One per judge)
    print("Initializing the 3 Explainers...")
    exp_aff = shap.GradientExplainer(model_aff, back_aff.to(device))
    exp_ecg = shap.GradientExplainer(model_ecg, back_ecg.to(device))
    exp_eda = shap.GradientExplainer(model_eda, back_eda.to(device))

    # 4. CALCULATING SHAP VALUES
    print("Calculating votes and explanations...")
    s_aff = np.array(exp_aff.shap_values(t_aff))
    s_ecg = np.array(exp_ecg.shap_values(t_ecg))
    s_eda = np.array(exp_eda.shap_values(t_eda))

    # 5. GENERATING PLOTS
    plots_dir = os.path.join(current_dir, 'plots_late_fusion')
    os.makedirs(plots_dir, exist_ok=True)

    for i in range(10):
        # Calculating individual votes
        with torch.no_grad():
            v_aff = 1 if torch.sigmoid(model_aff(t_aff[i:i+1])).item() > 0.5 else 0
            v_ecg = 1 if torch.sigmoid(model_ecg(t_ecg[i:i+1])).item() > 0.5 else 0
            v_eda = 1 if torch.sigmoid(model_eda(t_eda[i:i+1])).item() > 0.5 else 0
        
        voto_finale = 1 if (v_aff + v_ecg + v_eda) >= 2 else 0
        label_vera = int(t_labels[i])

        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f"SHAP Analysis LATE FUSION - Patient {i+1}\nTrue: {label_vera} | Final Verdict: {voto_finale}", 
                     fontsize=16, fontweight='bold')

        # Data for the loop
        data_list = [
            (t_aff[i].cpu().numpy().flatten(), np.squeeze(s_aff[i]).flatten(), v_aff, "AFFECT"),
            (t_ecg[i].cpu().numpy().flatten(), np.squeeze(s_ecg[i]).flatten(), v_ecg, "ECG"),
            (t_eda[i].cpu().numpy().flatten(), np.squeeze(s_eda[i]).flatten(), v_eda, "EDA")
        ]

        legend_handles = []

        for ch, (signal, shaps, voto, name) in enumerate(data_list):
            ax = axes[ch]
            # Black Signal
            line, = ax.plot(signal, color='black', alpha=0.6, linewidth=1.5, label='Real Signal')
            if ch == 0: legend_handles.append(line)

            # SHAP bars (multiplied by 10 for visibility as in early)
            colors = ['red' if float(val) > 0 else 'blue' for val in shaps]
            ax.bar(range(WINDOW_SIZE), shaps * 10, color=colors, width=1.0, alpha=0.7, edgecolor='none')
            
            ax.set_title(f"Judge {name} | Individual Vote: {voto}", fontsize=12, loc='left', fontweight='bold')
            ax.set_ylabel("SHAP Impact")
            ax.grid(True, alpha=0.2, linestyle='--')

        # Legenda
        red_p = mpatches.Rectangle((0,0),1,1, color='red', alpha=0.7, label='Pushes towards Positive (1)')
        blue_p = mpatches.Rectangle((0,0),1,1, color='blue', alpha=0.7, label='Pushes towards Negative (0)')
        legend_handles.extend([red_p, blue_p])
        fig.legend(handles=legend_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))

        plt.tight_layout(rect=[0, 0.05, 1, 0.94])
        save_path = os.path.join(plots_dir, f'shap_late_patient_{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Judges analysis saved: {save_path}")

if __name__ == "__main__":
    esegui_shap_fase2_late()

"""
=====================================================================================
INTERPRETATION GUIDE (LATE FUSION - THE COURT)
=====================================================================================

Here we do not see a single mind, but three distinct opinions.

1. THE CONFLICT OF THE JUDGES:
   - It is possible that the ECG Judge has all BLUE bars (vote 0) and the EDA Judge
     has all RED bars (vote 1). In this case, SHAP shows you
     exactly what convinced one and what misled the other.

2. THE STRENGTH OF THE VOTE:
   - If a judge votes 1 (Positive) but their SHAP bars are very low,
     it means it is a "weak" vote, almost uncertain.
   - If the bars are very high, the judge is convinced of their position.

3. COMPARISON WITH EARLY FUSION:
   - Note if the importance points (the colored bars) are in the same places as in
     Early Fusion. Often in Late Fusion the models are "simpler" and
     look at fewer details compared to Early Fusion, explaining why
     the overall accuracy is slightly lower.
=====================================================================================
"""