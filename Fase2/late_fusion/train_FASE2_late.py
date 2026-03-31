"""
=====================================================================================
Module: train_fase2_late.py
Project: ML Emotions - Phase 2 (Late Fusion Training)

Description:
Trains 3 1D-CNN networks (Affect, ECG, EDA) in parallel.
Saves each model independently based on its own MACRO F1-SCORE.
=====================================================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score # Import per la metrica God Tier

from config_FASE2_late import (BATCH_SIZE, LEARNING_RATE, EPOCHS, 
                               MODEL_SAVE_PATH_LATE_AFFECT, 
                               MODEL_SAVE_PATH_LATE_ECG, 
                               MODEL_SAVE_PATH_LATE_EDA)
from dataset_FASE2_late import PopaneDatasetLateFusion
from model_FASE2_late import UnimodalCNN 

def train_late_fusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting LATE FUSION training (3 Models)! Device: {device}")

    train_dataset = PopaneDatasetLateFusion(split_type="train")
    val_dataset = PopaneDatasetLateFusion(split_type="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 1. INITIALIZE THE 3 EXPERTS
    model_affect = UnimodalCNN().to(device)
    model_ecg = UnimodalCNN().to(device)
    model_eda = UnimodalCNN().to(device)
    
    print("Calculating weights...")
    num_positives, num_negatives = 0, 0
    for t_aff, t_ecg, t_eda, labels in train_loader:
        num_positives += labels.sum().item()
        num_negatives += (labels == 0).sum().item()

    weight_value = num_negatives / num_positives if num_positives > 0 else 1.0
    pos_weight = torch.tensor([weight_value]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 2. THREE OPTIMIZERS WITH WEIGHT DECAY
    opt_affect = optim.Adam(model_affect.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    opt_ecg = optim.Adam(model_ecg.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    opt_eda = optim.Adam(model_eda.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # 3. THREE INDEPENDENT SCHEDULERS
    sch_affect = optim.lr_scheduler.ReduceLROnPlateau(opt_affect, mode='min', factor=0.5, patience=4)
    sch_ecg = optim.lr_scheduler.ReduceLROnPlateau(opt_ecg, mode='min', factor=0.5, patience=4)
    sch_eda = optim.lr_scheduler.ReduceLROnPlateau(opt_eda, mode='min', factor=0.5, patience=4)

    # VARIABLES FOR EARLY STOPPING AND F1-SCORE RECORDS
    best_f1_aff, best_f1_ecg, best_f1_eda = 0.0, 0.0, 0.0
    
    # We do not put a real "brutal" Early Stopping that blocks the for loop (the 'break'),
    # because if the ECG finishes learning but the Affect is still rising, we must let it finish!

    print(f"\nStarting training for {EPOCHS} Epochs...\n")
    
    for epoch in range(EPOCHS):
        model_affect.train(); model_ecg.train(); model_eda.train()
        
        for batch_idx, (t_aff, t_ecg, t_eda, labels) in enumerate(train_loader):
            t_aff, t_ecg, t_eda, labels = t_aff.to(device), t_ecg.to(device), t_eda.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            # AFFECT
            opt_affect.zero_grad()
            out_aff = model_affect(t_aff)
            loss_aff = criterion(out_aff, labels)
            loss_aff.backward()
            opt_affect.step()

            # ECG
            opt_ecg.zero_grad()
            out_ecg = model_ecg(t_ecg)
            loss_ecg = criterion(out_ecg, labels)
            loss_ecg.backward()
            opt_ecg.step()

            # EDA
            opt_eda.zero_grad()
            out_eda = model_eda(t_eda)
            loss_eda = criterion(out_eda, labels)
            loss_eda.backward()
            opt_eda.step()

            if (batch_idx + 1) % 500 == 0:
                print(f"   -> Processed batch {batch_idx + 1}/{len(train_loader)}")

        # --- VALIDATION PHASE ---
        model_affect.eval(); model_ecg.eval(); model_eda.eval()
        v_loss_aff, v_loss_ecg, v_loss_eda, total_val = 0.0, 0.0, 0.0, 0
        
        # Separate lists for the predictions of the 3 judges
        preds_aff, preds_ecg, preds_eda = [], [], []
        true_labels = []

        with torch.no_grad():
            for t_aff, t_ecg, t_eda, labels in val_loader:
                t_aff, t_ecg, t_eda, labels = t_aff.to(device), t_ecg.to(device), t_eda.to(device), labels.to(device)
                labels = labels.unsqueeze(1)

                out_aff = model_affect(t_aff)
                out_ecg = model_ecg(t_ecg)
                out_eda = model_eda(t_eda)

                v_loss_aff += criterion(out_aff, labels).item() * labels.size(0)
                v_loss_ecg += criterion(out_ecg, labels).item() * labels.size(0)
                v_loss_eda += criterion(out_eda, labels).item() * labels.size(0)
                total_val += labels.size(0)
                
                # Saving predictions (0 or 1) to calculate F1
                preds_aff.extend((torch.sigmoid(out_aff) > 0.5).float().cpu().numpy())
                preds_ecg.extend((torch.sigmoid(out_ecg) > 0.5).float().cpu().numpy())
                preds_eda.extend((torch.sigmoid(out_eda) > 0.5).float().cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        # Calculate Averages
        if total_val > 0:
            v_loss_aff /= total_val
            v_loss_ecg /= total_val
            v_loss_eda /= total_val
            
            # Calculate the 3 Independent F1-Scores
            f1_aff = f1_score(true_labels, preds_aff, average='macro')
            f1_ecg = f1_score(true_labels, preds_ecg, average='macro')
            f1_eda = f1_score(true_labels, preds_eda, average='macro')
        else:
            v_loss_aff, v_loss_ecg, v_loss_eda = float('inf'), float('inf'), float('inf')
            f1_aff, f1_ecg, f1_eda = 0.0, 0.0, 0.0

        print(f"Epoch [{epoch+1}/{EPOCHS}] | VAL F1-MACRO -> Affect: {f1_aff:.4f} | ECG: {f1_ecg:.4f} | EDA: {f1_eda:.4f}")

        # SCHEDULER STEP (We leave it on the loss, because the loss is more mathematically continuous)
        sch_affect.step(v_loss_aff)
        sch_ecg.step(v_loss_ecg)
        sch_eda.step(v_loss_eda)

        # =========================================================
        # INDEPENDENT SAVING BASED ON F1-SCORE 
        # =========================================================
        if f1_aff > best_f1_aff:
            best_f1_aff = f1_aff
            torch.save(model_affect.state_dict(), MODEL_SAVE_PATH_LATE_AFFECT)
            
        if f1_ecg > best_f1_ecg:
            best_f1_ecg = f1_ecg
            torch.save(model_ecg.state_dict(), MODEL_SAVE_PATH_LATE_ECG)
            
        if f1_eda > best_f1_eda:
            best_f1_eda = f1_eda
            torch.save(model_eda.state_dict(), MODEL_SAVE_PATH_LATE_EDA)

    print("\nTraining Completed! The 3 best judges (Models) have been saved to disk!")
    print(f"Final F1-Macro Records -> Affect: {best_f1_aff:.4f} | ECG: {best_f1_ecg:.4f} | EDA: {best_f1_eda:.4f}")

if __name__ == "__main__":
    train_late_fusion()