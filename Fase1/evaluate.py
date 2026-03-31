"""
=====================================================================================
Modulo: evaluate.py
Descrizione: Script standalone per calcolare e stampare tutte le metriche finali
(Accuracy, F1-Score, AUC-ROC, Report e Matrice) caricando il modello già addestrato.
Stampa tutto a terminale e SOLO ALLA FINE salva il blocco dei risultati in output_fase1.txt.
=====================================================================================
"""
import torch
import numpy as np
import os
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
        for i, (inputs, labels) in enumerate(test_loader):
            # Stampiamo a terminale il progresso in tempo reale
            print(f"\r⏳ Elaborazione del batch {i+1}...su {len(test_loader)}", end="")
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
            
    print("\n\nAnalisi completata. Calcolo delle metriche in corso...")

    # 4. CALCOLO DELLE METRICHE
    y_true = np.array(tutte_le_label).astype(int)
    y_pred = np.array(tutte_le_predizioni).astype(int)
    y_prob = np.array(tutte_le_probabilita)
    
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float('nan') 
        
    class_report = classification_report(
        y_true, y_pred, 
        labels=[0, 1], 
        target_names=['Emozione Negativa (0)', 'Emozione Positiva (1)'],
        zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # 5. COSTRUZIONE DEL TESTO FINALE (IL "BLOCCHETTO" DEI RISULTATI)
    report_text = f"""==================================================
📊 RISULTATI FINALI DEL MODELLO
==================================================
✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)
✅ F1-Score:  {f1_macro:.4f} (Macro Average)
✅ AUC-ROC:   {auc:.4f}

--------------------------------------------------
📋 CLASSIFICATION REPORT DETTAGLIATO
--------------------------------------------------
{class_report}
--------------------------------------------------
🧩 MATRICE DI CONFUSIONE
--------------------------------------------------
Vero Negativo (TN): {cm[0][0]:<5} | Falso Positivo (FP): {cm[0][1]}
Falso Negativo (FN): {cm[1][0]:<5} | Vero Positivo (TP):  {cm[1][1]}
==================================================
"""

    # 6. STAMPA A TERMINALE (Così hai subito i risultati a vista)
    print("\n" + report_text)

    # 7. SALVATAGGIO SU FILE IN MODO SICURO
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(current_dir, 'output_fase1.txt')
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
            
        print(f"✅ Report salvato con successo nel file: {output_path}")
        
    except Exception as e:
        print(f"\n⚠️ ATTENZIONE: Impossibile salvare il file di testo ({e}).")
        print("Tranquillo, i risultati sono comunque stampati qui sopra! Nessun dato perso.")

if __name__ == "__main__":
    valuta_modello()