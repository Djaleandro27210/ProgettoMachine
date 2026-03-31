"""
=====================================================================================
Module: evaluate.py
Description: The "Court" of Late Fusion. Loads the 3 trained models,
performs Majority Voting and calculates the final metrics.
Prints everything to terminal and ONLY AT THE END saves the results in output_fase2_late.txt.
=====================================================================================
"""
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from config_FASE2_late import BATCH_SIZE, MODEL_SAVE_PATH_LATE_AFFECT, MODEL_SAVE_PATH_LATE_ECG, MODEL_SAVE_PATH_LATE_EDA
from dataset_FASE2_late import PopaneDatasetLateFusion
from model_FASE2_late import UnimodalCNN


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(path, model_class, device):
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def _vote_from_logits(logits):
    votes = torch.sigmoid(logits).squeeze()
    if votes.dim() == 0:
        votes = votes.unsqueeze(0)
    return (votes > 0.5).int()


def _majority_vote(votes_list):
    sum_votes = sum(votes_list)
    return (sum_votes >= 2).int()


def _compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=['Negative (0)', 'Positive (1)'],
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return accuracy, f1_macro, report, cm


def _render_report(accuracy, f1_macro, class_report, cm):
    return f"""==================================================
LATE FUSION RESULTS (MAJORITY VOTING)
==================================================
Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)
F1-Score:  {f1_macro:.4f} (Macro Average)
*(Note: AUC-ROC is not calculated in majority voting Late Fusion, because we get clear votes, not probabilities!)*

--------------------------------------------------
CLASSIFICATION REPORT
--------------------------------------------------
{class_report}
--------------------------------------------------
CONFUSION MATRIX
--------------------------------------------------
True Negative (TN): {cm[0][0]:<5} | False Positive (FP): {cm[0][1]}
False Negative (FN): {cm[1][0]:<5} | True Positive (TP):  {cm[1][1]}
==================================================
"""


def _save_report(report_text, filename='output_fase2_late.txt'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    return output_path


def evaluate_late_fusion():
    device = _get_device()
    print(f"Starting Late Fusion evaluation (majority voting). Device: {device}")

    test_dataset = PopaneDatasetLateFusion(split_type='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    try:
        model_aff = _load_model(MODEL_SAVE_PATH_LATE_AFFECT, UnimodalCNN, device)
        model_ecg = _load_model(MODEL_SAVE_PATH_LATE_ECG, UnimodalCNN, device)
        model_eda = _load_model(MODEL_SAVE_PATH_LATE_EDA, UnimodalCNN, device)
        print('All 3 models loaded successfully!')
    except Exception as e:
        print(f' Error loading models: {e}')
        return

    y_true = []
    y_pred = []

    print('\nStarting court hearings (Test Set Analysis)...')

    with torch.no_grad():
        for i, (t_aff, t_ecg, t_eda, labels) in enumerate(test_loader):
            print(f"Processing batch {i+1}/{len(test_loader)}", end='\r')
            t_aff, t_ecg, t_eda = t_aff.to(device), t_ecg.to(device), t_eda.to(device)

            out_aff = model_aff(t_aff)
            out_ecg = model_ecg(t_ecg)
            out_eda = model_eda(t_eda)

            vote_aff = _vote_from_logits(out_aff)
            vote_ecg = _vote_from_logits(out_ecg)
            vote_eda = _vote_from_logits(out_eda)

            pred_batch = _majority_vote([vote_aff, vote_ecg, vote_eda])

            y_pred.extend(pred_batch.cpu().numpy())
            y_true.extend(labels.numpy())

    print('\n\nCalculating metrics...')

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    accuracy, f1_macro, class_report, cm = _compute_metrics(y_true, y_pred)
    report_text = _render_report(accuracy, f1_macro, class_report, cm)

    print('\n' + report_text)

    try:
        saved_path = _save_report(report_text)
        print(f'Report saved successfully in file: {saved_path}')
    except Exception as e:
        print(f"\nWARNING: unable to save text file ({e}).")
        print('Results are still printed in console.')


if __name__ == '__main__':
    evaluate_late_fusion()