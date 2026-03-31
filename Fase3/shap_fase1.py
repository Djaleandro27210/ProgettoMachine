"""
=====================================================================================
Module: shap_fase1.py
Project: ML Emotions - Phase 3 (Explainable AI on ECG)

Description:
Final script for SHAP analysis. Explains how the CNN interprets the ECG.
=====================================================================================
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# =========================================================
# 1. PATH FIXES - TO FIND OTHER PHASES
# =========================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) # Project root
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src", "Fase1")) # Path to Phase 1

try:
    from config_FASE1 import BATCH_SIZE, MODEL_SAVE_PATH, WINDOW_SIZE
    from dataset_FASE1 import PopaneDataset
    from model_FASE1 import Emotion1DCNN
    import shap
    print("Modules loaded successfully!")
except ImportError as e:
    print(f"Critical error: Unable to find Phase 1 files. Error: {e}")
    sys.exit(1)

def esegui_shap_fase1():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting SHAP Analysis. Device: {device}")

    # 1. MODEL AND DATA LOADING
    test_dataset = PopaneDataset(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    model = Emotion1DCNN().to(device)
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model not found in {MODEL_SAVE_PATH}")
        return
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Phase 1 model loaded!") 
    

    # 2. DATA PREPARATION FOR SHAP
    data_iterator = iter(test_loader)
    background_inputs, _ = next(data_iterator)
    background_inputs = background_inputs.to(device)

    test_inputs, test_labels = next(data_iterator)
    test_inputs = test_inputs[:10].to(device) # We analyze the first 10 patients
    test_labels = test_labels[:10].cpu().numpy()

    # 3. SHAP VALUES CALCULATION
    print("Initializing GradientExplainer...")
    explainer = shap.GradientExplainer(model, background_inputs)
    
    print("Calculating feature importance (SHAP values)...")
    shap_values = explainer.shap_values(test_inputs)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Conversion for plotting
    test_inputs_np = test_inputs.cpu().numpy()
    shap_values_np = np.array(shap_values)

    # 4. PLOTS FOLDER MANAGEMENT (Absolute Path)
    plots_dir = os.path.join(current_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    print(f"Destination folder: {plots_dir}")

    # 5. GRAPH GENERATION
    print("Generating plots...")
    
    for i in range(10):
        # Data extraction and cleaning (Fix for TypeError)
        signal = test_inputs_np[i, 0, :].flatten()
        shaps = np.squeeze(shap_values_np[i, 0, :]).flatten()
        
        label_vera = int(test_labels[i])
        
        with torch.no_grad():
            output = model(test_inputs[i:i+1])
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0

        # Figure Creation
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"SHAP Analysis Patient {i+1}\nTrue: {label_vera} | Predicted: {pred} (Prob: {prob:.2f})", fontsize=14, fontweight='bold')

        # Subplot 1: ECG Signal
        ax1.plot(signal, color='black', linewidth=1.2, label='ECG Signal')
        ax1.set_title("ECG Waveform (Normalized)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)

        # Subplot 2: SHAP Values
        # We define colors: Red pushes towards 1 (Positive), Blue pushes towards 0 (Negative)
        colors = ['red' if float(val) > 0 else 'blue' for val in shaps]
        ax2.bar(range(len(shaps)), shaps, color=colors, width=1.0, edgecolor='none')
        ax2.set_title("SHAP Contribution (Red: towards Positive | Blue: towards Negative)")
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("SHAP Impact")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Saving and verification
        save_path = os.path.join(plots_dir, f'shap_fase1_paziente_{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        if os.path.exists(save_path):
            print(f"[OK] Saved: {save_path}")
        else:
            print(f"[ERROR] Failed to save in: {save_path}")

    print(f"\nANALYSIS COMPLETED! Find the files here: {plots_dir}")

if __name__ == "__main__":
    esegui_shap_fase1()
    """
=====================================================================================
GUIDE TO INTERPRETING SHAP GRAPHS (XAI - EXPLAINABLE AI)
=====================================================================================

These graphs allow you to "open the black box" of the 1D-CNN, showing which 
points of the ECG signal influenced the model's decision.

1. THE MEANING OF COLORS:
   - RED (SHAP > 0): The model interpreted these milliseconds as indicators
     of a POSITIVE emotion (Class 1). The higher the bar, the stronger the influence.
   - BLUE (SHAP < 0): The model interpreted these milliseconds as indicators
     of a NEGATIVE emotion (Class 0). The lower the bar, the more "convinced" the model is.

2. WHAT TO OBSERVE IN THE ECG WAVE:
   - QRS COMPLEX (The high peak): It is the most important part. If it is red, the network 
     associates the strength/speed of ventricular contraction with well-being. If it is blue,
     it detects anomalies in the peak that it associates with stress or discomfort.
   - T INTERVAL (The hump after the peak): Represents heart relaxation. 
     Significant contributions here indicate that the model is analyzing 
     'repolarization', a process very sensitive to the autonomic nervous system.
   - ISOELECTRIC LINE (The flat parts): Represent the rhythm. If the model 
     colors here, it is analyzing HRV (Heart Rate Variability).

3. MATHEMATICAL LOGIC:
   If the sum of the 'reds' exceeds that of the 'blues', the model 
   predicts a positive emotion.

4. ERROR ANALYSIS (False Positives/Negatives):
   In graphs where the model makes mistakes (e.g. True: 0, Predicted: 1), SHAP will reveal 
   which artifacts or morphologies of the signal have 'deceived' the network, providing 
   fundamental evidence to discuss the limits of the unimodal system.
=====================================================================================
"""