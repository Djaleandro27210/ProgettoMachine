"""
=====================================================================================
Modulo: test_fase2_late.py
Progetto: ML Emozioni - Fase 2 (Test Late Fusion)

Descrizione:
Test finale per la strategia Late Fusion.
Carica i 3 modelli indipendenti (Affect, ECG, EDA), fa fare una previsione a ciascuno,
e combina i risultati usando la logica del MAJORITY VOTING (Voto a maggioranza).
=====================================================================================
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# IMPORTIAMO I MODULI DELLA FASE 2 LATE
from config_FASE2_late import (BATCH_SIZE, 
                               MODEL_SAVE_PATH_LATE_AFFECT, 
                               MODEL_SAVE_PATH_LATE_ECG, 
                               MODEL_SAVE_PATH_LATE_EDA)
from dataset_FASE2_late import PopaneDatasetLateFusion
from model_FASE2_late import UnimodalCNN

def test_late_fusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Inizio Test LATE FUSION (Majority Voting)! Dispositivo: {device}")

    # 1. Caricamento Test Set Separato
    test_dataset = PopaneDatasetLateFusion(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Inizializzazione dei 3 Esperti
    model_affect = UnimodalCNN().to(device)
    model_ecg = UnimodalCNN().to(device)
    model_eda = UnimodalCNN().to(device)
    
   # try:
    #    model_affect.load_state_dict(torch.load(MODEL_SAVE_PATH_LATE_AFFECT, map_location=device, weights_only=True))
     #   model_ecg.load_state_dict(torch.load(MODEL_SAVE_PATH_LATE_ECG, map_location=device, weights_only=True))
      #  model_eda.load_state_dict(torch.load(MODEL_SAVE_PATH_LATE_EDA, map_location=device, weights_only=True))
      #  print("✅ Tutti e 3 i modelli (Affect, ECG, EDA) caricati con successo!")
   # except Exception as e:
    #    print(f"❌ Errore nel caricamento dei modelli: {e}")
    #    return
        
    model_affect.eval(); model_ecg.eval(); model_eda.eval()

    all_final_preds = []
    all_labels = []

    print(f"Inizio il Tribunale su {len(test_dataset)} fettine. Attendi la sentenza...")
    
    with torch.no_grad():
        for batch_idx, (t_aff, t_ecg, t_eda, labels) in enumerate(test_loader):
            
            t_aff, t_ecg, t_eda, labels = t_aff.to(device), t_ecg.to(device), t_eda.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            # A. Chiediamo l'opinione ai 3 esperti
            out_aff = model_affect(t_aff)
            out_ecg = model_ecg(t_ecg)
            out_eda = model_eda(t_eda)

            # B. Trasformiamo le opinioni in Voti Binari (0 o 1)
            # Usiamo .int() così possiamo sommarli matematicamente
            vote_aff = (torch.sigmoid(out_aff) > 0.5).int()
            vote_ecg = (torch.sigmoid(out_ecg) > 0.5).int()
            vote_eda = (torch.sigmoid(out_eda) > 0.5).int()

            # C. IL MAJORITY VOTING (La Magia)
            # Sommiamo i voti: se la somma è 2 o 3, significa che la maggioranza ha detto "Positivo" (1)
            # Se la somma è 0 o 1, la maggioranza ha detto "Negativo" (0)
            total_votes = vote_aff + vote_ecg + vote_eda
            final_decision = (total_votes >= 2).int()
            
            all_final_preds.extend(final_decision.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 100 == 0:
                print(f"   -> Giudicati {batch_idx + 1}/{len(test_loader)} batch")

    # 3. Metriche Finali
    print("\n📊 --- REPORT FINALE LATE FUSION (Voto a Maggioranza) ---")
    
    all_labels = np.array(all_labels).astype(int)
    all_final_preds = np.array(all_final_preds).astype(int)
    
    print(classification_report(
        all_labels, 
        all_final_preds, 
        labels=[0, 1], 
        target_names=['Negativo (0)', 'Positivo (1)'], 
        zero_division=0
    ))
    
    print("\n🧩 --- MATRICE DI CONFUSIONE LATE FUSION ---")
    cm = confusion_matrix(all_labels, all_final_preds, labels=[0, 1])
    
    print(f"Veri Negativi (TN): {cm[0][0]} | Falsi Positivi (FP): {cm[0][1]}")
    print(f"Falsi Negativi (FN): {cm[1][0]} | Veri Positivi (TP): {cm[1][1]}")

if __name__ == "__main__":
    test_late_fusion()