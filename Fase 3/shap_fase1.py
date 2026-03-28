"""
=====================================================================================
Modulo: analyze_fase1.py
Progetto: ML Emozioni - Fase 3 (Explainable AI su ECG)

Descrizione:
Utilizza SHAP (GradientExplainer) per interpretare le decisioni della 1D-CNN.
Spiega QUALI parti dell'onda ECG (quali dei 1000 campioni) spingono la rete
a decidere per l'emozione Positiva o Negativa.
Salva i grafici nella cartella 'plots'.
=====================================================================================
"""
import os
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Importa i moduli della Fase 1
# ⚠️ Assicurati che python trovi i file (potrebbe servire aggiungere la cartella madre al path)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config_FASE1 import BATCH_SIZE, MODEL_SAVE_PATH, WINDOW_SIZE
from dataset_FASE1 import PopaneDataset
from model_FASE1 import Emotion1DCNN

def esegui_shap_fase1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Avvio Analisi SHAP Fase 1. Dispositivo: {device}")

    # 1. PREPARAZIONE DATI E MODELLO
    # Usiamo il Test Set per estrarre sia il background che i pazienti da analizzare
    test_dataset = PopaneDataset(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True) # Batch da 100 per SHAP

    model = Emotion1DCNN().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
        print("✅ Modello Fase 1 caricato con successo!")
    except Exception as e:
        print(f"❌ Errore nel caricamento del modello: {e}")
        return
    model.eval()

    # 2. SELEZIONE DEI DATI PER SHAP
    # SHAP ha bisogno di un "Background" (dati di riferimento per capire cos'è un ECG normale)
    print("⏳ Estrazione dei dati di Background per SHAP...")
    data_iterator = iter(test_loader)
    
    # Prendiamo 100 esempi casuali come base di paragone
    background_inputs, _ = next(data_iterator)
    background_inputs = background_inputs.to(device)

    # Prendiamo 5 esempi specifici che vogliamo "Spiegare" per la tesi
    test_inputs, test_labels = next(data_iterator)
    test_inputs = test_inputs[:5].to(device)
    test_labels = test_labels[:5].cpu().numpy()

    # 3. INIZIALIZZAZIONE DI SHAP (GradientExplainer è ottimo per le PyTorch CNN)
    print("🧠 Inizializzazione di SHAP GradientExplainer (potrebbe richiedere un po')...")
    explainer = shap.GradientExplainer(model, background_inputs)

    # Calcoliamo i valori SHAP per i 5 pazienti scelti
    print("🔬 Calcolo dell'importanza delle feature sui pazienti di test...")
    # shape attesa dei valori SHAP: [5, 1, 1000]
    shap_values = explainer.shap_values(test_inputs)
    
    # SHAP a volte restituisce una lista (a seconda della versione), la standardizziamo:
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Convertiamo tutto in Numpy per disegnare i grafici
    test_inputs_np = test_inputs.cpu().numpy()
    shap_values_np = np.array(shap_values)

    # 4. CREAZIONE DELLA CARTELLA PER I PLOTS
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 5. DISEGNO DEI GRAFICI PER LA TESI (ECG Waveform + SHAP Colors)
    print("🎨 Generazione dei grafici in corso...")
    
    for i in range(5):
        signal = test_inputs_np[i, 0, :] # Il segnale ECG vero e proprio (1000 punti)
        shaps = shap_values_np[i, 0, :]  # Quanto ogni punto ha contato per la decisione
        label_vera = int(test_labels[i])
        
        # Calcoliamo la probabilità predetta dal modello per questo paziente
        with torch.no_grad():
            output = model(test_inputs[i:i+1])
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0

        # Disegniamo la figura
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle(f"Analisi SHAP Paziente {i+1}\nReale: {label_vera} | Predetto: {pred} (Prob: {prob:.2f})", fontsize=14)

        # Plot 1: Il segnale ECG originale
        ax1.plot(signal, color='black', linewidth=1)
        ax1.set_title("Segnale ECG Originale (Z-Score Normalized)")
        ax1.set_ylabel("Ampiezza")
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Plot 2: I Valori SHAP (Il colore fa capire chi spinge verso cosa)
        # Rosso = spinge per Emozione Positiva (1), Blu = spinge per Negativa (0)
        ax2.bar(range(WINDOW_SIZE), shaps, color=['red' if x > 0 else 'blue' for x in shaps], width=1.0)
        ax2.set_title("Impronta SHAP: Rosso = Spinge verso Positivo (1) | Blu = Spinge verso Negativo (0)")
        ax2.set_xlabel("Campionamento nel Tempo (Millisecondi)")
        ax2.set_ylabel("Valore SHAP")
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        
        # Salvataggio
        save_path = os.path.join(plots_dir, f'shap_fase1_paziente_{i+1}.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    print(f"\n🎉 Finito! Ho generato 5 grafici esplicativi altissima risoluzione nella cartella:\n{plots_dir}")
    print("Usali direttamente nella tua tesi per spiegare come 'ragiona' la rete.")

if __name__ == "__main__":
    esegui_shap_fase1()