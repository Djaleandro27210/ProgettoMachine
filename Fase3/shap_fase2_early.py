"""
shap_fase2_early.py
Standalone script for SHAP analysis on the Early Fusion architecture.
Generates multimodal visual explanations with real signals overlaid in black.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader

# 1. Path routing to locate Phase 2 Early Fusion modules
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) 
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src", "Fase2", "early_fusion"))

try:
    from config_FASE2_early import BATCH_SIZE, MODEL_SAVE_PATH_FASE2, WINDOW_SIZE
    from dataset_FASE2_early import PopaneDatasetMultimodal
    from model_FASE2_early import MultimodalEarlyFusionCNN
    import shap
    print("[INFO] Early Fusion modules loaded successfully.")
except ImportError as e:
    print(f"[ERROR] Failed to locate Phase 2 Early files. Error: {e}")
    sys.exit(1)


def execute_shap_early_fusion() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing SHAP Early Fusion Analysis on: {device}")

    # 1. Load Model and Data
    test_dataset = PopaneDatasetMultimodal(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)

    model = MultimodalEarlyFusionCNN().to(device)
    if not os.path.exists(MODEL_SAVE_PATH_FASE2):
        print(f"[ERROR] Model weights not found at {MODEL_SAVE_PATH_FASE2}")
        return
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH_FASE2, map_location=device, weights_only=True))
    model.eval()

    # 2. Prepare Data for SHAP
    data_iterator = iter(test_loader)
    background_inputs, _ = next(data_iterator)
    background_inputs = background_inputs.to(device)

    test_inputs, test_labels = next(data_iterator)
    test_inputs = test_inputs[:10].to(device) 
    test_labels = test_labels[:10].cpu().numpy()

    # 3. Calculate SHAP values
    print("[INFO] Initializing GradientExplainer...")
    explainer = shap.GradientExplainer(model, background_inputs)
    
    print("[INFO] Computing SHAP values (Multimodal)...")
    shap_values = explainer.shap_values(test_inputs)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    test_inputs_np = test_inputs.cpu().numpy()
    shap_values_np = np.array(shap_values)

    # 4. Manage Plot Directory
    plots_dir = os.path.join(current_dir, 'plots_early_fusion')
    os.makedirs(plots_dir, exist_ok=True)

    # 5. Plot Generation
    sensor_names = ["AFFECT (Face/Posture)", "ECG (Heart)", "EDA (Sweat)"]
    print("[INFO] Generating plots...")
    
    for i in range(10):
        true_label = int(test_labels[i])
        with torch.no_grad():
            output = model(test_inputs[i:i+1])
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0

        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f"SHAP EARLY FUSION Analysis - Patient {i+1}\nTrue: {true_label} | Predicted: {pred} (Prob: {prob:.2f})", 
                     fontsize=16, fontweight='bold')

        legend_handles = []

        # Iterate over the 3 multimodal channels
        for channel in range(3): 
            ax = axes[channel]
            signal = test_inputs_np[i, channel, :].flatten()
            shaps = shap_values_np[i, channel, :].flatten()
            
            # --- REAL SIGNAL IN BLACK ---
            line_real, = ax.plot(signal, color='black', alpha=0.6, linewidth=1.5, label='Real Signal')
            if channel == 0: 
                legend_handles.append(line_real)
            
            # --- SHAP BARS ---
            colors = ['red' if float(val) > 0 else 'blue' for val in shaps]
            ax.bar(range(WINDOW_SIZE), shaps * 10, color=colors, width=1.0, alpha=0.7, edgecolor='none')
            
            ax.set_title(f"Sensor: {sensor_names[channel]}", fontsize=12, loc='left', fontweight='bold')
            ax.set_ylabel("SHAP Impact")
            ax.grid(True, alpha=0.2, linestyle='--')

        # Custom Legend Proxies
        red_proxy = mpatches.Rectangle((0,0),1,1, color='red', alpha=0.7, label='Pushes towards Positive (1)')
        blue_proxy = mpatches.Rectangle((0,0),1,1, color='blue', alpha=0.7, label='Pushes towards Negative (0)')
        legend_handles.extend([red_proxy, blue_proxy])
        
        fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=True, facecolor='white', bbox_to_anchor=(0.5, 0.01))

        axes[2].set_xlabel("Time (ms)")
        plt.tight_layout(rect=[0, 0.05, 1, 0.94])
        
        save_path = os.path.join(plots_dir, f'shap_early_patient_{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"    -> Saved: {save_path}")

    print(f"\n[INFO] Analysis complete. Plots exported to: {plots_dir}")


if __name__ == "__main__":
    execute_shap_early_fusion()


"""
=====================================================================================
📖 GUIDE TO INTERPRETING MULTIMODAL SHAP PLOTS (EARLY FUSION)
=====================================================================================

These plots deconstruct how the Multimodal CNN interprets the combination of 
three different physiological signals simultaneously.

1. UNDERSTANDING THE VISUALS:
   - BLACK LINE: The actual normalized waveform of the signal (Affect, ECG, or EDA).
   - RED BARS (SHAP > 0): Specific moments in the signal that pushed the network 
     to predict a POSITIVE emotion (Class 1).
   - BLUE BARS (SHAP < 0): Specific moments in the signal that pushed the network 
     to predict a NEGATIVE emotion (Class 0).

2. CHANNEL-SPECIFIC OBSERVATIONS:
   - AFFECT (Face/Posture): Watch for sharp spikes in the black line. Do the SHAP 
     values turn red during sudden movement, or blue during stillness?
   - ECG (Heart): Observe the QRS complexes (highest peaks) and T-waves. The model 
     often anchors its decisions on changes in heart rate variability or waveform 
     distortions during these segments.
   - EDA (Sweat): The Galvanic Skin Response is slower. Look for broader "waves" of 
     red or blue. An upward trend in EDA often correlates with physiological arousal.

3. MULTIMODAL SYNERGY:
   The true power of Early Fusion is observing *when* sensors agree or disagree.
   For example, you might see the ECG pushing towards Negative (Blue) due to an 
   elevated heart rate, while the Affect sensor simultaneously pushes towards 
   Positive (Red) due to facial movement. The model weighs these competing 
   signals to make its final prediction.
=====================================================================================
"""