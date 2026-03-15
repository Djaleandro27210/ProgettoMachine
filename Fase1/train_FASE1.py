"""
=====================================================================================
Modulo: train.py
Progetto: ML Emozioni (Popane Dataset) - Fase 2 (Addestramento)

Descrizione:
Questo script è il "Training Loop". Gestisce l'apprendimento della Rete Neurale:
1. Carica i dati di Train e Validation.
2. Calcola le previsioni (Forward pass).
3. Misura l'errore tramite la Loss Function (BCEWithLogitsLoss).
4. Corregge i pesi della rete tramite l'Optimizer (Adam).
5. Salva il modello migliore su disco.
=====================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importiamo le nostre "creature" e i parametri dal config
from config_FASE1 import BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_SAVE_PATH
from dataset_FASE1 import PopaneDataset
from model_FASE1 import Emotion1DCNN

def train_model():
    # 1. SETUP DEL DISPOSITIVO (GPU vs CPU)
    # Se hai una scheda video NVIDIA, PyTorch andrà 10x più veloce.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inizio addestramento! Dispositivo utilizzato: {device}")

    # 2. PREPARAZIONE DEI DATI
    print("Caricamento dataset in corso...")
    train_dataset = PopaneDataset(split_type="train")
    val_dataset = PopaneDataset(split_type="val")

    # I DataLoader gestiscono la creazione dei "Batch" in automatico
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. INIZIALIZZAZIONE DELLA RETE E DEGLI STRUMENTI
    model = Emotion1DCNN().to(device) # Spostiamo la rete sulla GPU (se c'è)
    
    # --- MODIFICA PER IL BILANCIAMENTO DELLE CLASSI ---
    print("⚖️ Calcolo dei pesi per bilanciare le classi...")
    num_positives = 0
    num_negatives = 0
    
    # Facciamo un rapido giro sui dati di training per contare le classi
    for _, labels in train_loader:
        num_positives += labels.sum().item()
        num_negatives += (labels == 0).sum().item()

    # La formula per pos_weight nella BCE è: (numero di esempi negativi) / (numero di esempi positivi)
    # Se per qualche motivo non ci sono positivi (evitiamo la divisione per zero), impostiamo a 1.0
    weight_value = num_negatives / num_positives if num_positives > 0 else 1.0
    
    # Trasformiamo il valore in un tensore di PyTorch e lo spostiamo sul device corretto
    pos_weight = torch.tensor([weight_value]).to(device)
    print(f"📊 Esempi Negativi (0): {int(num_negatives)} | Esempi Positivi (1): {int(num_positives)}")
    print(f"⚖️ Moltiplicatore pos_weight calcolato: {weight_value:.2f}")

    # Loss Function: BCEWithLogitsLoss con le "multe" salatissime per gli errori sulla classe minoritaria!
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # --------------------------------------------------
    
    # Ottimizzatore: Adam (il migliore per iniziare, aggiusta i pesi in base all'errore)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Variabile per ricordarci qual è stata l'accuratezza migliore e salvare quel modello
    best_val_accuracy = 0.0

    # 4. IL CICLO DI ADDESTRAMENTO (THE LOOP)
    print(f"\nInizia il training per {EPOCHS} Epoche...\n")
    
    for epoch in range(EPOCHS):
        # --- FASE DI TRAINING ---
        model.train() # Diciamo alla rete "Guarda che ora si studia, accendi il Dropout!"
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # Iteriamo su tutti i batch del training set
        # AGGIUNTA VIP: Usiamo enumerate per contare a che batch siamo
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

            # --- IL NOSTRO CONTACHILOMETRI ---
            # Stampa un aggiornamento ogni 100 batch completati
            if (batch_idx + 1) % 100 == 0:
                print(f"   -> Sto faticando... Elaborato batch {batch_idx + 1}/{len(train_loader)}")

        # Calcolo medie di fine epoca per il Training
        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100

        # --- FASE DI VALIDATION ---
        model.eval() # Diciamo alla rete "Ora è un esame, non studiare e spegni il Dropout"
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        # torch.no_grad() spegne il calcolo delle correzioni (risparmia tantissima RAM) SERVE PER EVITARE OVERFITTING
        with torch.no_grad():
           # Aggiungiamo enumerate così batch_idx riparte da 0 per la validazione
            for batch_idx_val, (inputs, labels) in enumerate(val_loader):
                
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predictions == labels).sum().item()
                total_val += labels.size(0)
                
        # Calcolo medie di fine epoca per il Validation
        # AGGIUNGIAMO IL CONTROLLO ANTI-CRASH
        if total_val > 0:
            epoch_val_loss = running_val_loss / total_val
            epoch_val_acc = (correct_val / total_val) * 100
        else:
            epoch_val_loss = 0.0
            epoch_val_acc = 0.0
            print("⚠️ Attenzione: Il Validation Set sembra vuoto!")

        # Stampa dei risultati di questa epoca
        print(f"Epoca [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}%")

        # 5. SALVATAGGIO DEL MODELLO MIGLIORE
        # Se in questa epoca abbiamo battuto il record sul Validation Set, salviamo la rete!
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"⭐ Nuovo record! Modello salvato in {MODEL_SAVE_PATH}")

    print(f"\n🎉 Addestramento Completato! Miglior Accuratezza di Validazione: {best_val_accuracy:.2f}%")

if __name__ == "__main__":
    train_model()