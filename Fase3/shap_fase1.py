"""
shap_fase1.py
Standalone script for SHAP analysis (Explainable AI).
Visualizes how the 1D-CNN interprets the ECG signal for binary classification.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 1. Path routing to locate Phase 1 modules
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) 
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src", "Fase1")) 

try:
    from config_FASE1 import BATCH_SIZE, MODEL_SAVE_PATH, WINDOW_SIZE
    from dataset_FASE1 import PopaneDataset
    from model_FASE1 import Emotion1DCNN
    import shap
    print("[INFO] Required modules loaded successfully.")
except ImportError as e:
    print(f"[ERROR] Failed to locate Phase 1 files. Error: {e}")
    sys.exit(1)


def execute_shap_analysis() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing SHAP Analysis on: {device}")

    # 1. Load Model and Data
    test_dataset = PopaneDataset(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    model = Emotion1DCNN().to(device)
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"[ERROR] Model weights not found at {MODEL_SAVE_PATH}")
        return
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()
    print("[INFO] Phase 1 Model loaded.") 

    # 2. Prepare Data for SHAP
    data_iterator = iter(test_loader)
    background_inputs, _ = next(data_iterator)
    background_inputs = background_inputs.to(device)

    test_inputs, test_labels = next(data_iterator)
    test_inputs = test_inputs[:10].to(device)  # Analyze the first 10 patients
    test_labels = test_labels[:10].cpu().numpy()

    # 3. Calculate SHAP values
    print("[INFO] Initializing GradientExplainer...")
    explainer = shap.GradientExplainer(model, background_inputs)
    
    print("[INFO] Computing feature importance (SHAP values)...")
    shap_values = explainer.shap_values(test_inputs)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    test_inputs_np = test_inputs.cpu().numpy()
    shap_values_np = np.array(shap_values)

    # 4. Manage Plot Directory
    plots_dir = os.path.join(current_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    print(f"[INFO] Output directory configured: {plots_dir}")

    # 5. Plot Generation
    print("[INFO] Generating plots...")
    
    for i in range(10):
        # Extract and flatten the specific signal and its corresponding SHAP values
        signal = test_inputs_np[i, 0, :].flatten()
        shaps = np.squeeze(shap_values_np[i, 0, :]).flatten()
        
        true_label = int(test_labels[i])
        
        with torch.no_grad():
            output = model(test_inputs[i:i+1])
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0

        # Build Figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"SHAP Analysis - Patient {i+1}\nTrue: {true_label} | Predicted: {pred} (Prob: {prob:.2f})", 
                     fontsize=14, fontweight='bold')

        # Subplot 1: ECG Signal
        ax1.plot(signal, color='black', linewidth=1.2, label='ECG Signal')
        ax1.set_title("ECG Waveform (Normalized)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)

        # Subplot 2: SHAP Values
        # Red pushes towards 1 (Positive), Blue pushes towards 0 (Negative)
        colors = ['red' if float(val) > 0 else 'blue' for val in shaps]
        ax2.bar(range(len(shaps)), shaps, color=colors, width=1.0, edgecolor='none')
        ax2.set_title("SHAP Contribution (Red: towards Positive | Blue: towards Negative)")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("SHAP Impact")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save mechanism
        save_path = os.path.join(plots_dir, f'shap_fase1_patient_{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        if os.path.exists(save_path):
            print(f"    -> Saved: {save_path}")
        else:
            print(f"    -> [ERROR] Failed to save: {save_path}")

    print(f"\n[INFO] Analysis complete. Plots exported to: {plots_dir}")


if __name__ == "__main__":
    execute_shap_analysis()


"""
=====================================================================================
📖 GUIDE TO INTERPRETING SHAP PLOTS (XAI - EXPLAINABLE AI)
=====================================================================================

These plots allow us to "open the black box" of the 1D-CNN, revealing which specific 
points of the ECG signal influenced the model's decision.

1. THE MEANING OF COLORS:
   - RED (SHAP > 0): The model interpreted these milliseconds as indicators of a 
     POSITIVE emotion (Class 1). The taller the bar, the stronger the influence.
   - BLUE (SHAP < 0): The model interpreted these milliseconds as indicators of a 
     NEGATIVE emotion (Class 0). The lower the bar, the more "convinced" the model is.

2. WHAT TO LOOK FOR IN THE ECG WAVEFORM:
   - QRS COMPLEX (The tall peak): This is the most critical segment. If it is highlighted
     red, the network associates the strength/speed of ventricular contraction with 
     well-being. If blue, it detects peak anomalies associated with stress or discomfort.
   - T WAVE (The hump following the peak): Represents the relaxation of the heart. 
     Significant contributions here indicate the model is analyzing 'repolarization', 
     a process highly sensitive to the autonomic nervous system.
   - ISOELECTRIC LINE (The flat segments): Represents the rhythm. If the model 
     highlights these areas, it is likely analyzing HRV (Heart Rate Variability).

3. MATHEMATICAL LOGIC:
   If the sum of the 'red' values exceeds the sum of the 'blue' values, the model 
   predicts a positive emotion.

4. ERROR ANALYSIS (False Positives/Negatives):
   In plots where the model makes a mistake (e.g., True: 0, Predicted: 1), SHAP 
   reveals which artifacts or signal morphologies 'tricked' the network. This 
   provides fundamental evidence for discussing the limitations of the unimodal system.
=====================================================================================
"""