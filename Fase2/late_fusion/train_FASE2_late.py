"""
train_FASE3_late.py
Training loop for Phase 3 (Late Fusion).
Trains 3 parallel 1D-CNNs (Affect, ECG, EDA) independently.
Saves each model based on its individual peak Macro F1-Score.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from config_FASE2_late import (
    BATCH_SIZE, 
    LEARNING_RATE, 
    EPOCHS, 
    MODEL_SAVE_PATH_LATE_AFFECT, 
    MODEL_SAVE_PATH_LATE_ECG, 
    MODEL_SAVE_PATH_LATE_EDA,
    WEIGHT_DECAY
)
from dataset_FASE2_late import PopaneDatasetLateFusion
from model_FASE2_late import UnimodalCNN 


def calculate_class_weights(train_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Computes the positive class weight to offset dataset imbalance."""
    num_positives = 0
    num_negatives = 0
    
    for _, _, _, labels in train_loader:
        num_positives += labels.sum().item()
        num_negatives += (labels == 0).sum().item()

    weight_value = num_negatives / num_positives if num_positives > 0 else 1.0
    
    print(f"[INFO] Class distribution - Negative: {int(num_negatives)} | Positive: {int(num_positives)}")
    print(f"[INFO] Computed pos_weight multiplier: {weight_value:.4f}")
    
    return torch.tensor([weight_value]).to(device)


def train_late_fusion() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing Phase 3 (Late Fusion) Training on: {device}")

    print("[INFO] Loading datasets...")
    train_dataset = PopaneDatasetLateFusion(split_type="train")
    val_dataset = PopaneDatasetLateFusion(split_type="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 1. Initialize the 3 parallel models
    model_affect = UnimodalCNN().to(device)
    model_ecg = UnimodalCNN().to(device)
    model_eda = UnimodalCNN().to(device)
    
    pos_weight = calculate_class_weights(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 2. Three independent Optimizers
    opt_affect = optim.Adam(model_affect.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    opt_ecg = optim.Adam(model_ecg.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    opt_eda = optim.Adam(model_eda.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 3. Three independent Schedulers
    sch_affect = optim.lr_scheduler.ReduceLROnPlateau(opt_affect, mode='min', factor=0.5, patience=4)
    sch_ecg = optim.lr_scheduler.ReduceLROnPlateau(opt_ecg, mode='min', factor=0.5, patience=4)
    sch_eda = optim.lr_scheduler.ReduceLROnPlateau(opt_eda, mode='min', factor=0.5, patience=4)

    # F1-Score trackers for checkpointing
    best_f1_aff = 0.0
    best_f1_ecg = 0.0
    best_f1_eda = 0.0

    print(f"[INFO] Beginning parallel training loop for {EPOCHS} epochs...\n")
    
    for epoch in range(EPOCHS):
        
        # --- TRAINING PHASE ---
        model_affect.train()
        model_ecg.train()
        model_eda.train()
        
        for batch_idx, (t_aff, t_ecg, t_eda, labels) in enumerate(train_loader):
            t_aff = t_aff.to(device)
            t_ecg = t_ecg.to(device)
            t_eda = t_eda.to(device)
            labels = labels.to(device).unsqueeze(1)

            # Train Affect Model
            opt_affect.zero_grad()
            out_aff = model_affect(t_aff)
            loss_aff = criterion(out_aff, labels)
            loss_aff.backward()
            opt_affect.step()

            # Train ECG Model
            opt_ecg.zero_grad()
            out_ecg = model_ecg(t_ecg)
            loss_ecg = criterion(out_ecg, labels)
            loss_ecg.backward()
            opt_ecg.step()

            # Train EDA Model
            opt_eda.zero_grad()
            out_eda = model_eda(t_eda)
            loss_eda = criterion(out_eda, labels)
            loss_eda.backward()
            opt_eda.step()

            if (batch_idx + 1) % 500 == 0:
                print(f"    -> Processed batch {batch_idx + 1}/{len(train_loader)}")

        # --- VALIDATION PHASE ---
        model_affect.eval()
        model_ecg.eval()
        model_eda.eval()
        
        v_loss_aff, v_loss_ecg, v_loss_eda = 0.0, 0.0, 0.0
        total_val = 0
        
        preds_aff, preds_ecg, preds_eda = [], [], []
        true_labels = []

        with torch.no_grad():
            for t_aff, t_ecg, t_eda, labels in val_loader:
                t_aff = t_aff.to(device)
                t_ecg = t_ecg.to(device)
                t_eda = t_eda.to(device)
                labels = labels.to(device).unsqueeze(1)

                out_aff = model_affect(t_aff)
                out_ecg = model_ecg(t_ecg)
                out_eda = model_eda(t_eda)

                v_loss_aff += criterion(out_aff, labels).item() * labels.size(0)
                v_loss_ecg += criterion(out_ecg, labels).item() * labels.size(0)
                v_loss_eda += criterion(out_eda, labels).item() * labels.size(0)
                total_val += labels.size(0)
                
                preds_aff.extend((torch.sigmoid(out_aff) > 0.5).float().cpu().numpy())
                preds_ecg.extend((torch.sigmoid(out_ecg) > 0.5).float().cpu().numpy())
                preds_eda.extend((torch.sigmoid(out_eda) > 0.5).float().cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        if total_val > 0:
            v_loss_aff /= total_val
            v_loss_ecg /= total_val
            v_loss_eda /= total_val
            
            f1_aff = f1_score(true_labels, preds_aff, average='macro')
            f1_ecg = f1_score(true_labels, preds_ecg, average='macro')
            f1_eda = f1_score(true_labels, preds_eda, average='macro')
        else:
            v_loss_aff, v_loss_ecg, v_loss_eda = float('inf'), float('inf'), float('inf')
            f1_aff, f1_ecg, f1_eda = 0.0, 0.0, 0.0

        print(f"Epoch [{epoch+1}/{EPOCHS}] | VAL F1-MACRO -> Affect: {f1_aff:.4f} | ECG: {f1_ecg:.4f} | EDA: {f1_eda:.4f}")

        # Update Schedulers
        sch_affect.step(v_loss_aff)
        sch_ecg.step(v_loss_ecg)
        sch_eda.step(v_loss_eda)

        # --- INDEPENDENT CHECKPOINTING ---
        if f1_aff > best_f1_aff:
            best_f1_aff = f1_aff
            torch.save(model_affect.state_dict(), MODEL_SAVE_PATH_LATE_AFFECT)
            
        if f1_ecg > best_f1_ecg:
            best_f1_ecg = f1_ecg
            torch.save(model_ecg.state_dict(), MODEL_SAVE_PATH_LATE_ECG)
            
        if f1_eda > best_f1_eda:
            best_f1_eda = f1_eda
            torch.save(model_eda.state_dict(), MODEL_SAVE_PATH_LATE_EDA)

    print("\n[INFO] Parallel Training Completed.")
    print(f"[INFO] Final Peak F1-Macro -> Affect: {best_f1_aff:.4f} | ECG: {best_f1_ecg:.4f} | EDA: {best_f1_eda:.4f}")


if __name__ == "__main__":
    train_late_fusion()