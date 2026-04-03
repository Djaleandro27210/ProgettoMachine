"""
evaluate.py
Standalone evaluation script for Phase 1. 
Calculates and logs Accuracy, F1-Score, AUC-ROC, and Confusion Matrix.
"""

import os
import sys
from typing import Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix
)

from config_FASE1 import BATCH_SIZE, MODEL_SAVE_PATH
from dataset_FASE1 import PopaneDataset
from model_FASE1 import Emotion1DCNN


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluation initialized on: {device}")
    return device


def load_trained_model(device: torch.device) -> Emotion1DCNN:
    model = Emotion1DCNN().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
        print("[INFO] Model weights loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model. Ensure training completed. Error: {e}")
        sys.exit(1)
        
    model.eval()
    return model


def perform_inference(
    model: Emotion1DCNN, 
    test_loader: DataLoader, 
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    all_labels = []
    all_preds = []
    all_probs = []

    print("[INFO] Running inference on test set...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            print(f"\r[PROCESS] Batch {batch_idx + 1}/{len(test_loader)}", end="", flush=True)
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze() 
            preds = (probs > 0.5).float()
            
            # Handle single-element batch dimension collapse
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
                preds = preds.unsqueeze(0)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n[INFO] Inference complete.")
    
    return (
        np.array(all_labels).astype(int), 
        np.array(all_preds).astype(int), 
        np.array(all_probs)
    )


def generate_report(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> str:
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan') 
        
    clf_report = classification_report(
        y_true, y_pred, 
        labels=[0, 1], 
        target_names=['Negative Emotion (0)', 'Positive Emotion (1)'],
        zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return f"""==================================================
FINAL MODEL EVALUATION RESULTS
==================================================
Accuracy:  {accuracy:.4f} ({accuracy * 100:.1f}%)
F1-Score:  {f1_macro:.4f} (Macro Average)
AUC-ROC:   {auc:.4f}

--------------------------------------------------
DETAILED CLASSIFICATION REPORT
--------------------------------------------------
{clf_report}
--------------------------------------------------
CONFUSION MATRIX
--------------------------------------------------
True Negative (TN):  {cm[0][0]:<5} | False Positive (FP): {cm[0][1]}
False Negative (FN): {cm[1][0]:<5} | True Positive (TP):  {cm[1][1]}
==================================================
"""


def save_report(report_text: str, filename: str = "output_fase1.txt") -> None:
    try:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[INFO] Report saved to: {output_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save report to disk. Error: {e}")


def evaluate_model() -> None:
    device = get_device()
    
    test_loader = DataLoader(PopaneDataset(split_type="test"), batch_size=BATCH_SIZE, shuffle=False)
    model = load_trained_model(device)
    
    y_true, y_pred, y_prob = perform_inference(model, test_loader, device)
    
    report_text = generate_report(y_true, y_pred, y_prob)
    print("\n" + report_text)
    save_report(report_text)


if __name__ == "__main__":
    evaluate_model()