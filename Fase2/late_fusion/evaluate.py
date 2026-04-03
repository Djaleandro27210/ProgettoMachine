"""
evaluate_FASE3_late.py
Standalone evaluation script for Phase 3 (Late Fusion).
Loads 3 independent models, performs Majority Voting, and logs final metrics.
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
    classification_report, 
    confusion_matrix
)

from config_FASE2_late import (
    BATCH_SIZE, 
    MODEL_SAVE_PATH_LATE_AFFECT, 
    MODEL_SAVE_PATH_LATE_ECG, 
    MODEL_SAVE_PATH_LATE_EDA
)
from dataset_FASE2_late import PopaneDatasetLateFusion
from model_FASE2_late import UnimodalCNN


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluation initialized on: {device}")
    return device


def load_trained_models(device: torch.device) -> Tuple[UnimodalCNN, UnimodalCNN, UnimodalCNN]:
    model_aff = UnimodalCNN().to(device)
    model_ecg = UnimodalCNN().to(device)
    model_eda = UnimodalCNN().to(device)
    
    try:
        model_aff.load_state_dict(torch.load(MODEL_SAVE_PATH_LATE_AFFECT, map_location=device, weights_only=True))
        model_ecg.load_state_dict(torch.load(MODEL_SAVE_PATH_LATE_ECG, map_location=device, weights_only=True))
        model_eda.load_state_dict(torch.load(MODEL_SAVE_PATH_LATE_EDA, map_location=device, weights_only=True))
        print("[INFO] All 3 Unimodal models loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load one or more models. Error: {e}")
        sys.exit(1)
        
    model_aff.eval()
    model_ecg.eval()
    model_eda.eval()
    
    return model_aff, model_ecg, model_eda


def perform_majority_voting(
    model_aff: UnimodalCNN,
    model_ecg: UnimodalCNN,
    model_eda: UnimodalCNN,
    test_loader: DataLoader, 
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    
    all_labels = []
    all_preds = []

    print("[INFO] Running Majority Voting inference on test set...")
    
    with torch.no_grad():
        for batch_idx, (t_aff, t_ecg, t_eda, labels) in enumerate(test_loader):
            print(f"\r[PROCESS] Batch {batch_idx + 1}/{len(test_loader)}", end="", flush=True)
            
            t_aff = t_aff.to(device)
            t_ecg = t_ecg.to(device)
            t_eda = t_eda.to(device)
            
            # Retrieve probabilities
            out_aff = model_aff(t_aff)
            out_ecg = model_ecg(t_ecg)
            out_eda = model_eda(t_eda)
            
            # Convert probabilities to binary votes (Threshold = 0.5)
            vote_aff = (torch.sigmoid(out_aff).squeeze() > 0.5).int()
            vote_ecg = (torch.sigmoid(out_ecg).squeeze() > 0.5).int()
            vote_eda = (torch.sigmoid(out_eda).squeeze() > 0.5).int()

            # Handle single-element batch dimension collapse
            if vote_aff.dim() == 0:
                vote_aff = vote_aff.unsqueeze(0)
                vote_ecg = vote_ecg.unsqueeze(0)
                vote_eda = vote_eda.unsqueeze(0)
            
            # Majority Voting: Sum votes. If >= 2, final prediction is 1, else 0.
            sum_votes = vote_aff + vote_ecg + vote_eda
            final_preds = (sum_votes >= 2).int()
            
            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("\n[INFO] Inference complete.")
    
    return np.array(all_labels).astype(int), np.array(all_preds).astype(int)


def generate_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    clf_report = classification_report(
        y_true, y_pred, 
        labels=[0, 1], 
        target_names=['Negative Emotion (0)', 'Positive Emotion (1)'],
        zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return f"""==================================================
FINAL EVALUATION RESULTS - LATE FUSION
==================================================
Accuracy:  {accuracy:.4f} ({accuracy * 100:.1f}%)
F1-Score:  {f1_macro:.4f} (Macro Average)
* Note: AUC-ROC is omitted in Majority Voting as predictions are hard votes, not probabilities.

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


def save_report(report_text: str, filename: str = "output_fase3_late.txt") -> None:
    try:
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[INFO] Report saved to: {output_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save report to disk. Error: {e}")


def evaluate_late_fusion() -> None:
    device = get_device()
    
    test_loader = DataLoader(PopaneDatasetLateFusion(split_type="test"), batch_size=BATCH_SIZE, shuffle=False)
    
    model_aff, model_ecg, model_eda = load_trained_models(device)
    
    y_true, y_pred = perform_majority_voting(model_aff, model_ecg, model_eda, test_loader, device)
    
    report_text = generate_report(y_true, y_pred)
    print("\n" + report_text)
    save_report(report_text)


if __name__ == "__main__":
    evaluate_late_fusion()