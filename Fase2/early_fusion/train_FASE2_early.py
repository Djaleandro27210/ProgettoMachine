"""
=====================================================================================
Modulo: train_FASE2_early.py
Progetto: ML Emozioni - Fase 2 (Addestramento Early Fusion)

Descrizione:
Script di addestramento per l'architettura Early Fusion a 3 canali.
Integrato con Early Stopping e Learning Rate Scheduler.
Salva il modello migliore su disco BASANDOSI SUL MACRO F1-SCORE.
=====================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score # <-- IMPORT FONDAMENTALE PER L'F1

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
    
    # --- Calcolo DINAMICO dei pesi ---
    print("⚖️ Calcolo dei pesi per bilanciare le classi in Early Fusion...")
    num_positives = 0
    num_negatives = 0
    
    for inputs, labels in train_loader:
        num_positives += labels.sum().item()
        num_negatives += (labels == 0).sum().item()

    weight_value = num_negatives / num_positives if num_positives > 0 else 1.0
    pos_weight = torch.tensor([weight_value]).to(device)
    
    print(f"📊 Esempi Negativi (0): {int(num_negatives)} | Esempi Positivi (1): {int(num_positives)}")
    print(f"⚖️ Moltiplicatore pos_weight calcolato: {weight_value:.4f}")

    # Loss e Ottimizzatore
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # =========================================================
    # INIZIALIZZAZIONE SCHEDULER E EARLY STOPPING (F1-SCORE)
    # =========================================================
    # Lo scheduler guarda ancora la Val Loss per capire se c'è stallo matematico
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    # L'Early Stopping e il Salvataggio ora guardano il Macro F1-Score
    patience_early_stopping = 10
    epochs_no_improve = 0
    best_val_f1 = 0.0 # Partiamo da un F1 di zero
    # =========================================================

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

            if (batch_idx + 1) % 500 == 0:
                print(f"   -> Elaborato batch {batch_idx + 1}/{len(train_loader)}")

        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100

        # --- FASE DI VALIDATION ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        # Liste per memorizzare predizioni e label per l'F1-Score
        all_val_preds = []
        all_val_labels = []

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
                
                # Aggiungiamo i dati alle liste per calcolare l'F1
                all_val_preds.extend(predictions.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        # Calcolo medie di fine epoca per il Validation
        if total_val > 0:
            epoch_val_loss = running_val_loss / total_val
            epoch_val_acc = (correct_val / total_val) * 100
            
            # CALCOLO DEL MACRO F1-SCORE
            epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        else:
            epoch_val_loss = float('inf')
            epoch_val_acc = 0.0
            epoch_val_f1 = 0.0
            print("⚠️ Attenzione: Il Validation Set sembra vuoto!")

        # Stampa dei risultati completi
        print(f"Epoca [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}% - Val F1-Macro: {epoch_val_f1:.4f}")

        # =========================================================
        # LOGICA DI SCHEDULING E SALVATAGGIO
        # =========================================================
        # 1. Il Cecchino: Aggiorna il Learning Rate guardando la Loss
        scheduler.step(epoch_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < LEARNING_RATE and epochs_no_improve == 0: 
             print(f"📉 [SCHEDULER] Il learning rate è stato abbassato a: {current_lr}")

        # 2. Controllo Early Stopping: L'F1-Macro è salito?
        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            epochs_no_improve = 0 
            # SALVA IL MODELLO SOLO QUANDO L'F1-SCORE È AL MASSIMO
            torch.save(model.state_dict(), MODEL_SAVE_PATH_FASE2)
            print(f"⭐ Nuovo record di F1-Macro ({best_val_f1:.4f})! Modello salvato.")
        else:
            epochs_no_improve += 1
            print(f"⚠️ L'F1-Macro non è migliorato da {epochs_no_improve} epoche.")

        # 3. Freno a mano basato sull'F1
        if epochs_no_improve >= patience_early_stopping:
            print(f"\n🛑 EARLY STOPPING INNESCATO ALL'EPOCA {epoch+1}!")
            print(f"Il modello ha smesso di bilanciare le prestazioni per {patience_early_stopping} epoche. Interrompo.")
            break 

    print(f"\n🎉 Addestramento Completato! Miglior Val F1-Macro: {best_val_f1:.4f}")

if __name__ == "__main__":
    train_early_fusion()