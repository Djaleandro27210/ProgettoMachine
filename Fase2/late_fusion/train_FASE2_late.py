"""
=====================================================================================
Modulo: train_fase2_late.py
Progetto: ML Emozioni - Fase 2 (Addestramento Late Fusion)

Descrizione:
Addestra 3 reti neurali INDIPENDENTI in parallelo (Affect, ECG, EDA).
Salva i 3 modelli migliori separatamente.
=====================================================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# IMPORTA I FILE GIUSTI DELLA FASE 2 LATE
from config_FASE2_late import (BATCH_SIZE, LEARNING_RATE, EPOCHS, 
                               MODEL_SAVE_PATH_LATE_AFFECT, 
                               MODEL_SAVE_PATH_LATE_ECG, 
                               MODEL_SAVE_PATH_LATE_EDA)
from dataset_FASE2_late import PopaneDatasetLateFusion
from model_FASE2_late import UnimodalCNN

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
    
    criterion = nn.BCEWithLogitsLoss()
    
    # 2. TRE OTTIMIZZATORI SEPARATI
    opt_affect = optim.Adam(model_affect.parameters(), lr=LEARNING_RATE)
    opt_ecg = optim.Adam(model_ecg.parameters(), lr=LEARNING_RATE)
    opt_eda = optim.Adam(model_eda.parameters(), lr=LEARNING_RATE)

    # 3. RECORD DA BATTERE PER OGNI MODELLO
    best_acc_affect, best_acc_ecg, best_acc_eda = 0.0, 0.0, 0.0

    print(f"\nInizia il training per {EPOCHS} Epoche...\n")
    
    for epoch in range(EPOCHS):
        model_affect.train(); model_ecg.train(); model_eda.train()
        
        corr_aff, corr_ecg, corr_eda, total_train = 0, 0, 0, 0

        for batch_idx, (t_aff, t_ecg, t_eda, labels) in enumerate(train_loader):
            # Spostiamo tutto su GPU/CPU
            t_aff, t_ecg, t_eda, labels = t_aff.to(device), t_ecg.to(device), t_eda.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            # --- TRAINING AFFECT ---
            opt_affect.zero_grad()
            out_aff = model_affect(t_aff)
            loss_aff = criterion(out_aff, labels)
            loss_aff.backward()
            opt_affect.step()
            corr_aff += ((torch.sigmoid(out_aff) > 0.5).float() == labels).sum().item()

            # --- TRAINING ECG ---
            opt_ecg.zero_grad()
            out_ecg = model_ecg(t_ecg)
            loss_ecg = criterion(out_ecg, labels)
            loss_ecg.backward()
            opt_ecg.step()
            corr_ecg += ((torch.sigmoid(out_ecg) > 0.5).float() == labels).sum().item()

            # --- TRAINING EDA ---
            opt_eda.zero_grad()
            out_eda = model_eda(t_eda)
            loss_eda = criterion(out_eda, labels)
            loss_eda.backward()
            opt_eda.step()
            corr_eda += ((torch.sigmoid(out_eda) > 0.5).float() == labels).sum().item()

            total_train += labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"   -> Elaborato batch {batch_idx + 1}/{len(train_loader)}")

        # --- FASE DI VALIDATION ---
        model_affect.eval(); model_ecg.eval(); model_eda.eval()
        v_corr_aff, v_corr_ecg, v_corr_eda, total_val = 0, 0, 0, 0

        with torch.no_grad():
            for t_aff, t_ecg, t_eda, labels in val_loader:
                t_aff, t_ecg, t_eda, labels = t_aff.to(device), t_ecg.to(device), t_eda.to(device), labels.to(device)
                labels = labels.unsqueeze(1)

                out_aff = model_affect(t_aff)
                out_ecg = model_ecg(t_ecg)
                out_eda = model_eda(t_eda)

                v_corr_aff += ((torch.sigmoid(out_aff) > 0.5).float() == labels).sum().item()
                v_corr_ecg += ((torch.sigmoid(out_ecg) > 0.5).float() == labels).sum().item()
                v_corr_eda += ((torch.sigmoid(out_eda) > 0.5).float() == labels).sum().item()
                total_val += labels.size(0)

        # Calcolo delle accuratezze di Validazione
        val_acc_aff = (v_corr_aff / total_val) * 100 if total_val > 0 else 0
        val_acc_ecg = (v_corr_ecg / total_val) * 100 if total_val > 0 else 0
        val_acc_eda = (v_corr_eda / total_val) * 100 if total_val > 0 else 0

        print(f"Epoca [{epoch+1}/{EPOCHS}] | VAL ACCURACIES -> Affect: {val_acc_aff:.2f}% | ECG: {val_acc_ecg:.2f}% | EDA: {val_acc_eda:.2f}%")

        # SALVATAGGIO INDIPENDENTE
        if val_acc_aff > best_acc_affect:
            best_acc_affect = val_acc_aff
            torch.save(model_affect.state_dict(), MODEL_SAVE_PATH_LATE_AFFECT)
            
        if val_acc_ecg > best_acc_ecg:
            best_acc_ecg = val_acc_ecg
            torch.save(model_ecg.state_dict(), MODEL_SAVE_PATH_LATE_ECG)
            
        if val_acc_eda > best_acc_eda:
            best_acc_eda = val_acc_eda
            torch.save(model_eda.state_dict(), MODEL_SAVE_PATH_LATE_EDA)

    print("\n🎉 Training Completato! Modelli salvati con successo!")

if __name__ == "__main__":
    train_late_fusion()