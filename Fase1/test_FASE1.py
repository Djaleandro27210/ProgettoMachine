"""
=====================================================================================
Modulo: test.py (Versione con Freno a Mano)
Descrizione: Test finale sul Test Set per verificare le metriche (Classification 
Report e Matrice di Confusione). Freno attivato per test rapido.
=====================================================================================
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from config_FASE1 import BATCH_SIZE, MODEL_SAVE_PATH
from dataset_FASE1 import PopaneDataset
from model_FASE1 import Emotion1DCNN

def test_model_full():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Inizio Test Finale  Dispositivo utilizzato: {device}")

    test_dataset = PopaneDataset(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Emotion1DCNN().to(device)
  #  try:
   #     model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    #    print("✅ Modello caricato con successo!")
    #except Exception as e:
     #   print(f"❌ Errore nel caricamento del modello: {e}")
      #  return
        
    model.eval()

    all_preds = []
    all_labels = []

    print(f"Inizio analisi...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            
     

            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            outputs = model(inputs)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            print(f"   -> Elaborato batch {batch_idx + 1}")

    print("\n📊 --- REPORT FINALE ---")
    
   # # Assicuriamoci che i numeri siano interi per Scikit-Learn (0 e 1)
   # all_labels = np.array(all_labels).astype(int)
   # all_preds = np.array(all_preds).astype(int)
   # 
   # # AGGIUNTA VIP: labels=[0, 1] forza la lettura di entrambe le classi
   # print(classification_report(
   #     all_labels, 
   #     all_preds, 
   #     labels=[0, 1], 
   #     target_names=['Negativo (0)', 'Positivo (1)'], 
   #     zero_division=0
   # ))
   # 
   # print("\n🧩 --- MATRICE DI CONFUSIONE ---")
   # cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
   # 
   # # Ora la matrice sarà sempre 2x2, garantito!
   # print(f"Veri Negativi (TN): {cm[0][0]} | Falsi Positivi (FP): {cm[0][1]}")
   # print(f"Falsi Negativi (FN): {cm[1][0]} | Veri Positivi (TP): {cm[1][1]}")
#
if __name__ == "__main__":
   test_model_full()
