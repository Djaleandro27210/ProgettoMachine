"""
=====================================================================================
Modulo: test_FASE2_early.py
Progetto: ML Emozioni - Fase 2 (Test Early Fusion)

Descrizione:
Test finale sul Test Set per l'architettura Early Fusion Multimodale.
Carica il modello salvato e stampa Classification Report e Matrice di Confusione.
=====================================================================================
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# IMPORTIAMO I MODULI DELLA FASE 2
from config_FASE2_early import BATCH_SIZE, MODEL_SAVE_PATH_FASE2
from dataset_FASE2_early import PopaneDatasetMultimodal
from model_FASE2_early import MultimodalEarlyFusionCNN

def test_early_fusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Inizio Test EARLY FUSION! Dispositivo utilizzato: {device}")

    # 1. Caricamento Test Set Multimodale
    test_dataset = PopaneDatasetMultimodal(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Inizializzazione Rete e Caricamento Pesi
    model = MultimodalEarlyFusionCNN().to(device)
    
   # try:
     #    model.load_state_dict(torch.load(MODEL_SAVE_PATH_FASE2, map_location=device, weights_only=True))
      #  print(f"✅ Modello multimodale caricato con successo da: {MODEL_SAVE_PATH_FASE2}")
   # except Exception as e:
    #    print(f"❌ Errore nel caricamento del modello: {e}")
     #   return
        
    model.eval()

    all_preds = []
    all_labels = []

    print(f"Inizio analisi su {len(test_dataset)} fettine multimodali. Attendi...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            outputs = model(inputs)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 100 == 0:
                print(f"   -> Elaborati {batch_idx + 1}/{len(test_loader)} batch")

    # 3. Metriche Finali
    print("\n📊 --- REPORT FINALE EARLY FUSION ---")
    
    all_labels = np.array(all_labels).astype(int)
    all_preds = np.array(all_preds).astype(int)
    
    # zero_division=0 e labels=[0,1] per evitare crash
    print(classification_report(
        all_labels, 
        all_preds, 
        labels=[0, 1], 
        target_names=['Negativo (0)', 'Positivo (1)'], 
        zero_division=0
    ))
    
    print("\n🧩 --- MATRICE DI CONFUSIONE EARLY FUSION ---")
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    
    print(f"Veri Negativi (TN): {cm[0][0]} | Falsi Positivi (FP): {cm[0][1]}")
    print(f"Falsi Negativi (FN): {cm[1][0]} | Veri Positivi (TP): {cm[1][1]}")

if __name__ == "__main__":
    test_early_fusion()