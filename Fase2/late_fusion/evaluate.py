"""
=====================================================================================
Modulo: evaluate_late.py
Descrizione: Il "Tribunale" della Late Fusion. Carica i 3 modelli addestrati,
effettua il Voto a Maggioranza (Majority Voting) e calcola le metriche finali.
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

def valuta_late_fusion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Avvio Valutazione LATE FUSION (Voto a Maggioranza). Dispositivo: {device}")

    test_dataset = PopaneDatasetLateFusion(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 1. CARICHIAMO I 3 GIUDICI
    model_aff = UnimodalCNN().to(device)
    model_ecg = UnimodalCNN().to(device)
    model_eda = UnimodalCNN().to(device)
    
    try:
        model_aff.load_state_dict(torch.load(MODEL_SAVE_PATH_LATE_AFFECT, map_location=device, weights_only=True))
        model_ecg.load_state_dict(torch.load(MODEL_SAVE_PATH_LATE_ECG, map_location=device, weights_only=True))
        model_eda.load_state_dict(torch.load(MODEL_SAVE_PATH_LATE_EDA, map_location=device, weights_only=True))
        print("✅ Tutti e 3 i modelli caricati con successo!")
    except Exception as e:
        print(f"❌ Errore nel caricamento dei modelli: {e}")
        return
        
    model_aff.eval(); model_ecg.eval(); model_eda.eval()

    tutte_le_label = []
    tutte_le_predizioni_finali = []

    print("\nInizio udienze del Tribunale (Analisi Test Set)...")
    
    with torch.no_grad():
        for i, (t_aff, t_ecg, t_eda, labels) in enumerate(test_loader):
            print(f"\r⏳ Elaborazione del batch {i+1}...su {len(test_loader)}", end="")
            t_aff, t_ecg, t_eda = t_aff.to(device), t_ecg.to(device), t_eda.to(device)
            
            # I 3 GIUDICI ESPRIMONO LA LORO PROBABILITÀ
            out_aff = model_aff(t_aff)
            out_ecg = model_ecg(t_ecg)
            out_eda = model_eda(t_eda)
            
            # TRASFORMIAMO LE PROBABILITÀ IN VOTI (0 o 1)
            vote_aff = (torch.sigmoid(out_aff).squeeze() > 0.5).int()
            vote_ecg = (torch.sigmoid(out_ecg).squeeze() > 0.5).int()
            vote_eda = (torch.sigmoid(out_eda).squeeze() > 0.5).int()

            # Gestione del caso in cui c'è un solo elemento nel batch (lo squeeze toglie troppo)
            if vote_aff.dim() == 0:
                vote_aff, vote_ecg, vote_eda = vote_aff.unsqueeze(0), vote_ecg.unsqueeze(0), vote_eda.unsqueeze(0)
            
            # IL VOTO A MAGGIORANZA (Sommiamo i voti. Se la somma è >= 2, vince 1, altrimenti 0)
            somma_voti = vote_aff + vote_ecg + vote_eda
            pred_finale = (somma_voti >= 2).int()
            
            tutte_le_predizioni_finali.extend(pred_finale.cpu().numpy())
            tutte_le_label.extend(labels.numpy())

    print("\n\nCalcolo delle metriche in corso...")

    y_true = np.array(tutte_le_label).astype(int)
    y_pred = np.array(tutte_le_predizioni_finali).astype(int)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    class_report = classification_report(y_true, y_pred, labels=[0, 1], target_names=['Negativa (0)', 'Positiva (1)'], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report_text = f"""==================================================
📊 RISULTATI LATE FUSION (VOTO A MAGGIORANZA)
==================================================
✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)
✅ F1-Score:  {f1_macro:.4f} (Macro Average)
*(Nota: L'AUC-ROC non si calcola nella Late Fusion a maggioranza, perché otteniamo voti netti, non probabilità!)*

--------------------------------------------------
📋 CLASSIFICATION REPORT
--------------------------------------------------
{class_report}
--------------------------------------------------
🧩 MATRICE DI CONFUSIONE
--------------------------------------------------
Vero Negativo (TN): {cm[0][0]:<5} | Falso Positivo (FP): {cm[0][1]}
Falso Negativo (FN): {cm[1][0]:<5} | Vero Positivo (TP):  {cm[1][1]}
==================================================
"""
    print("\n" + report_text)

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(current_dir, 'output_fase2_late.txt')
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"✅ Report salvato in: {output_path}")
    except Exception as e:
        print(f"\n⚠️ Impossibile salvare il file txt ({e}).")

if __name__ == "__main__":
    valuta_late_fusion()