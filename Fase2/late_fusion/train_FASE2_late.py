"""
=====================================================================================
Modulo: train_fase2_late.py
Progetto: ML Emozioni - Fase 2 (Addestramento Late Fusion)
=====================================================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config_FASE2_late import (BATCH_SIZE, LEARNING_RATE, EPOCHS, 
                               MODEL_SAVE_PATH_LATE_AFFECT, 
                               MODEL_SAVE_PATH_LATE_ECG, 
                               MODEL_SAVE_PATH_LATE_EDA)
from dataset_FASE2_late import PopaneDatasetLateFusion
from model_FASE2_late import UnimodalCNN # Assicurati che il nome file sia corretto!

def train_late_fusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inizio addestramento LATE FUSION (3 Modelli)! Dispositivo: {device}")

    train_dataset = PopaneDatasetLateFusion(split_type="train")
    val_dataset = PopaneDatasetLateFusion(split_type="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 1. INIZIALIZZIAMO I 3 "ESPERTI"
    model_affect = UnimodalCNN().to(device)
    model_ecg = UnimodalCNN().to(device)
    model_eda = UnimodalCNN().to(device)
    
    print("⚖️ Calcolo dei pesi...")
    num_positives, num_negatives = 0, 0
    for t_aff, t_ecg, t_eda, labels in train_loader:
        num_positives += labels.sum().item()
        num_negatives += (labels == 0).sum().item()

    weight_value = num_negatives / num_positives if num_positives > 0 else 1.0
    pos_weight = torch.tensor([weight_value]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 2. TRE OTTIMIZZATORI CON WEIGHT DECAY (1e-3 per stroncare l'overfitting)
    opt_affect = optim.Adam(model_affect.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    opt_ecg = optim.Adam(model_ecg.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    opt_eda = optim.Adam(model_eda.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # 3. TRE SCHEDULER INDIPENDENTI
    sch_affect = optim.lr_scheduler.ReduceLROnPlateau(opt_affect, mode='min', factor=0.5, patience=4)
    sch_ecg = optim.lr_scheduler.ReduceLROnPlateau(opt_ecg, mode='min', factor=0.5, patience=4)
    sch_eda = optim.lr_scheduler.ReduceLROnPlateau(opt_eda, mode='min', factor=0.5, patience=4)

    best_loss_aff, best_loss_ecg, best_loss_eda = float('inf'), float('inf'), float('inf')

    print(f"\nInizia il training per {EPOCHS} Epoche...\n")
    
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
                print(f"   -> Elaborato batch {batch_idx + 1}/{len(train_loader)}")

        # --- FASE DI VALIDATION ---
        model_affect.eval(); model_ecg.eval(); model_eda.eval()
        v_loss_aff, v_loss_ecg, v_loss_eda, total_val = 0.0, 0.0, 0.0, 0

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

        v_loss_aff /= total_val
        v_loss_ecg /= total_val
        v_loss_eda /= total_val

        print(f"Epoca [{epoch+1}/{EPOCHS}] | VAL LOSS -> Affect: {v_loss_aff:.4f} | ECG: {v_loss_ecg:.4f} | EDA: {v_loss_eda:.4f}")

        # SCHEDULER STEP
        sch_affect.step(v_loss_aff)
        sch_ecg.step(v_loss_ecg)
        sch_eda.step(v_loss_eda)

        # SALVATAGGIO INDIPENDENTE SULLA LOSS
        if v_loss_aff < best_loss_aff:
            best_loss_aff = v_loss_aff
            torch.save(model_affect.state_dict(), MODEL_SAVE_PATH_LATE_AFFECT)
            
        if v_loss_ecg < best_loss_ecg:
            best_loss_ecg = v_loss_ecg
            torch.save(model_ecg.state_dict(), MODEL_SAVE_PATH_LATE_ECG)
            
        if v_loss_eda < best_loss_eda:
            best_loss_eda = v_loss_eda
            torch.save(model_eda.state_dict(), MODEL_SAVE_PATH_LATE_EDA)

    print("\n🎉 Training Completato! I 3 modelli migliori sono stati salvati!")

if __name__ == "__main__":
    train_late_fusion()