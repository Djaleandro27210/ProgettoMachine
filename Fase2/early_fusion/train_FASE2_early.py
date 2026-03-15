"""
=====================================================================================
Modulo: train_FASE2_early.py
Progetto: ML Emozioni - Fase 2 (Addestramento Early Fusion)

Descrizione:
Script di addestramento per l'architettura Early Fusion a 3 canali.
=====================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ⚠️ ATTENZIONE: Importiamo dai file della FASE 2!
from config_FASE2_early import BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_SAVE_PATH_FASE2
from dataset_FASE2_early import PopaneDatasetMultimodal
from model_FASE2_early import MultimodalEarlyFusionCNN

def train_early_fusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inizio addestramento EARLY FUSION! Dispositivo: {device}")

    # 1. Caricamento Dataset Multimodale (3 canali)
    print("Caricamento dataset multimodale in corso...")
    train_dataset = PopaneDatasetMultimodal(split_type="train")
    val_dataset = PopaneDatasetMultimodal(split_type="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Inizializzazione Rete Multimodale
    model = MultimodalEarlyFusionCNN().to(device)
    
    # IMPORTANTE: Ti consiglio di aggiungere pos_weight se le classi sono sbilanciate!
    # Se non lo usi, metti solo nn.BCEWithLogitsLoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    print(f"\nInizia il training per {EPOCHS} Epoche...\n")
    
    for epoch in range(EPOCHS):
        # --- FASE DI TRAINING ---
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_train += (predictions == labels).sum().item()
            total_train += labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"   -> Elaborato batch {batch_idx + 1}/{len(train_loader)}")

        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100

        # --- FASE DI VALIDATION ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_idx_val, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predictions == labels).sum().item()
                total_val += labels.size(0)

        if total_val > 0:
            epoch_val_loss = running_val_loss / total_val
            epoch_val_acc = (correct_val / total_val) * 100
        else:
            epoch_val_loss = 0.0
            epoch_val_acc = 0.0

        print(f"Epoca [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            # Salviamo usando il percorso della Fase 2!
            torch.save(model.state_dict(), MODEL_SAVE_PATH_FASE2)
            print(f"⭐ Nuovo record! Modello salvato in {MODEL_SAVE_PATH_FASE2}")

    print(f"\n🎉 Addestramento Completato! Miglior Accuratezza di Validazione: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    train_early_fusion()