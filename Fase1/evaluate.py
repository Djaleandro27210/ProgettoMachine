"""
=====================================================================================
Modulo: evaluate.py
Descrizione: Script standalone per calcolare e stampare tutte le metriche finali
(Accuracy, F1-Score, AUC-ROC, Report e Matrice) caricando il modello già addestrato.
=====================================================================================
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    classification_report, 
    confusion_matrix
)

# Importa i tuoi moduli (assicurati che i nomi corrispondano ai tuoi file)
from config_FASE1 import BATCH_SIZE, MODEL_SAVE_PATH
from dataset_FASE1 import PopaneDataset
from model_FASE1 import Emotion1DCNN

def valuta_modello():
    # 1. SETUP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Avvio Valutazione. Dispositivo: {device}")

    # Carichiamo SOLO il Test Set
    test_dataset = PopaneDataset(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. CARICAMENTO DEL MODELLO GIA' ADDESTRATO
    model = Emotion1DCNN().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
        print("✅ Modello caricato con successo dal disco! Nessun riaddestramento necessario.")
    except Exception as e:
        print(f"❌ Impossibile trovare o caricare il modello. Hai fatto girare train.py? Errore: {e}")
        return
        
    model.eval() # Modalità esame

    # Liste per salvare i risultati
    tutte_le_label = []
    tutte_le_predizioni = []
    tutte_le_probabilita = []

    print("\nInizio analisi dei dati di test...")
    
    # 3. INFERENZA (Calcolo delle predizioni sui dati nuovi)
    with torch.no_grad(): # Niente calcolo dei gradienti = velocissimo
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Passaggio in avanti
            outputs = model(inputs)
            
            # Calcolo probabilità e predizioni (0 o 1)
            probs = torch.sigmoid(outputs).squeeze() # Squeeze per evitare problemi di dimensioni
            preds = (probs > 0.5).float()
            
            # Se il batch ha un solo elemento, squeeze() toglie troppe dimensioni, gestiamolo:
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
                preds = preds.unsqueeze(0)
            
            tutte_le_probabilita.extend(probs.cpu().numpy())
            tutte_le_predizioni.extend(preds.cpu().numpy())
            tutte_le_label.extend(labels.cpu().numpy())
            
    # 4. CALCOLO DELLE METRICHE
    print("\n" + "="*50)
    print("📊 RISULTATI FINALI DEL MODELLO")
    print("="*50)
    
    y_true = np.array(tutte_le_label).astype(int)
    y_pred = np.array(tutte_le_predizioni).astype(int)
    y_prob = np.array(tutte_le_probabilita)
    
    # La tua Triade di Metriche
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro') # Media tra le due classi
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan') # Nel caso rarissimo in cui il test set abbia solo 1 classe
        
    print(f"✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"✅ F1-Score:  {f1_macro:.4f} (Macro Average)")
    print(f"✅ AUC-ROC:   {auc:.4f}")
    
    print("\n" + "-"*50)
    print("📋 CLASSIFICATION REPORT DETTAGLIATO")
    print("-"*50)
    print(classification_report(
        y_true, y_pred, 
        labels=[0, 1], 
        target_names=['Emozione Negativa (0)', 'Emozione Positiva (1)'],
        zero_division=0
    ))
    
    print("\n" + "-"*50)
    print("🧩 MATRICE DI CONFUSIONE")
    print("-"*50)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"Vero Negativo (TN): {cm[0][0]:<5} | Falso Positivo (FP): {cm[0][1]}")
    print(f"Falso Negativo (FN): {cm[1][0]:<5} | Vero Positivo (TP):  {cm[1][1]}")
    print("="*50 + "\n")

if __name__ == "__main__":
    valuta_modello()