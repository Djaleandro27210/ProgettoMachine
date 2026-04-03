"""
shap_FASE3_late.py
Standalone script for SHAP analysis on the Late Fusion architecture.
Interrogates the 3 independent "Judges" (Affect, ECG, EDA) to explain their 
individual decisions prior to the Majority Vote.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

# 1. Path routing to locate Late Fusion modules
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) 
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src", "Fase2", "late_fusion"))

try:
    from config_FASE2_late import (
        BATCH_SIZE, WINDOW_SIZE,
        MODEL_SAVE_PATH_LATE_AFFECT, 
        MODEL_SAVE_PATH_LATE_ECG, 
        MODEL_SAVE_PATH_LATE_EDA
    )
    from dataset_FASE2_late import PopaneDatasetLateFusion
    from model_FASE2_late import UnimodalCNN
    import shap
    print("[INFO] Late Fusion modules loaded successfully.")
except ImportError as e:
    print(f"[ERROR] Critical import failure: {e}")
    sys.exit(1)


def execute_shap_late_fusion() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing SHAP Late Fusion Analysis (3 Judges) on: {device}")

    # 1. Load the 3 independent models
    test_dataset = PopaneDatasetLateFusion(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    def load_model(path: str) -> UnimodalCNN:
        m = UnimodalCNN().to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        return m

    model_aff = load_model(MODEL_SAVE_PATH_LATE_AFFECT)
    model_ecg = load_model(MODEL_SAVE_PATH_LATE_ECG)
    model_eda = load_model(MODEL_SAVE_PATH_LATE_EDA)

    # 2. Prepare Data for SHAP
    data_iterator = iter(test_loader)
    back_aff, back_ecg, back_eda, _ = next(data_iterator)
    
    t_aff, t_ecg, t_eda, t_labels = next(data_iterator)
    t_aff = t_aff[:10].to(device)
    t_ecg = t_ecg[:10].to(device)
    t_eda = t_eda[:10].to(device)
    t_labels = t_labels[:10].numpy()

    # 3. Create 3 independent Explainers
    print("[INFO] Initializing 3 separate GradientExplainers...")
    exp_aff = shap.GradientExplainer(model_aff, back_aff.to(device))
    exp_ecg = shap.GradientExplainer(model_ecg, back_ecg.to(device))
    exp_eda = shap.GradientExplainer(model_eda, back_eda.to(device))

    # 4. Calculate SHAP values
    print("[INFO] Computing individual votes and SHAP explanations...")
    shaps_aff = np.array(exp_aff.shap_values(t_aff))
    shaps_ecg = np.array(exp_ecg.shap_values(t_ecg))
    shaps_eda = np.array(exp_eda.shap_values(t_eda))

    # 5. Generate Plots
    plots_dir = os.path.join(current_dir, 'plots_late_fusion')
    os.makedirs(plots_dir, exist_ok=True)

    print("[INFO] Generating plots...")
    for i in range(10):
        # Calculate individual predictions
        with torch.no_grad():
            v_aff = 1 if torch.sigmoid(model_aff(t_aff[i:i+1])).item() > 0.5 else 0
            v_ecg = 1 if torch.sigmoid(model_ecg(t_ecg[i:i+1])).item() > 0.5 else 0
            v_eda = 1 if torch.sigmoid(model_eda(t_eda[i:i+1])).item() > 0.5 else 0
        
        # Calculate Majority Vote
        final_vote = 1 if (v_aff + v_ecg + v_eda) >= 2 else 0
        true_label = int(t_labels[i])

        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f"SHAP LATE FUSION Analysis - Patient {i+1}\nTrue: {true_label} | Final Majority Verdict: {final_vote}", 
                     fontsize=16, fontweight='bold')

        # Structure data for easy iteration
        data_list = [
            (t_aff[i].cpu().numpy().flatten(), np.squeeze(shaps_aff[i]).flatten(), v_aff, "AFFECT"),
            (t_ecg[i].cpu().numpy().flatten(), np.squeeze(shaps_ecg[i]).flatten(), v_ecg, "ECG"),
            (t_eda[i].cpu().numpy().flatten(), np.squeeze(shaps_eda[i]).flatten(), v_eda, "EDA")
        ]

        legend_handles = []

        for ch, (signal, shaps, vote, name) in enumerate(data_list):
            ax = axes[ch]
            
            # Real Signal
            line, = ax.plot(signal, color='black', alpha=0.6, linewidth=1.5, label='Real Signal')
            if ch == 0: 
                legend_handles.append(line)

            # SHAP Bars (Scaled by 10 for visibility)
            colors = ['red' if float(val) > 0 else 'blue' for val in shaps]
            ax.bar(range(WINDOW_SIZE), shaps * 10, color=colors, width=1.0, alpha=0.7, edgecolor='none')
            
            ax.set_title(f"Judge: {name} | Individual Vote: {vote}", fontsize=12, loc='left', fontweight='bold')
            ax.set_ylabel("SHAP Impact")
            ax.grid(True, alpha=0.2, linestyle='--')

        # Legend Setup
        red_p = mpatches.Rectangle((0,0),1,1, color='red', alpha=0.7, label='Pushes towards Positive (1)')
        blue_p = mpatches.Rectangle((0,0),1,1, color='blue', alpha=0.7, label='Pushes towards Negative (0)')
        legend_handles.extend([red_p, blue_p])
        fig.legend(handles=legend_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.01))

        plt.tight_layout(rect=[0, 0.05, 1, 0.94])
        
        save_path = os.path.join(plots_dir, f'shap_late_patient_{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"    -> Saved: {save_path}")

    print(f"\n[INFO] Analysis complete. Plots exported to: {plots_dir}")


if __name__ == "__main__":
    execute_shap_late_fusion()


"""
=====================================================================================
📖 GUIDE TO INTERPRETING SHAP PLOTS (LATE FUSION - THE TRIBUNAL)
=====================================================================================

Unlike Early Fusion, we are not looking at a single neural network mind. We are 
looking at three distinct opinions.

1. THE CONFLICT OF THE JUDGES:
   - It is entirely possible for the ECG Judge to display all BLUE bars (voting 0) 
     while the EDA Judge displays all RED bars (voting 1). SHAP reveals exactly 
     which specific signal morphologies convinced one judge and tricked the other.

2. VOTE CONFIDENCE (STRENGTH):
   - If a judge votes 1 (Positive) but their SHAP bars are extremely short, it 
     indicates a "weak" or uncertain vote. 
   - Conversely, very tall bars indicate the model is highly confident in its 
     decision based on those specific time steps.

3. COMPARING TO EARLY FUSION:
   - Check if the points of high importance (the tallest colored bars) align with 
     the ones found in the Early Fusion analysis. Often, Late Fusion models are 
     "simpler" in their feature extraction and focus on fewer fine details than an 
     Early Fusion model, which partially explains why Late Fusion might have a 
     slightly lower overall accuracy.
=====================================================================================
"""