import os
import glob
import pandas as pd

# ==========================================
# 1. CONFIGURAZIONI INIZIALI
# ==========================================
# Percorsi delle cartelle (assicurati che esistano)
from config import RAW_DIR, PROCESSED_DIR, WINDOW_SIZE, EMOTION_MAP, INDEX_PATH

def create_dataset_index():
    print("Inizio scansione del dataset...")
    
    # Creiamo una lista vuota che conterrà tutte le "fettine" da dare in pasto alla rete
    index_data = []
    
    # Troviamo TUTTI i file .csv dentro RAW_DIR e le sue sottocartelle (es. study1, study2)
    # L'asterisco doppio (**) significa "cerca in tutte le sottocartelle"
    search_pattern = os.path.join(RAW_DIR, "**", "*.csv")
    all_csv_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Trovati {len(all_csv_files)} file CSV totali. Filtro in corso...")

    # ==========================================
    # 2. CICLO SUI FILE
    # ==========================================
    for filepath in all_csv_files:
        # Estraiamo solo il nome del file dal percorso completo (es. da "data/raw/study1/185_anger.csv" a "185_anger.csv")
        filename = os.path.basename(filepath)
        
        # Togliamo l'estensione .csv (diventa "185_anger")
        name_without_ext = filename.replace('.csv', '')
        
        # Dividiamo il nome usando l'underscore "_" per isolare l'emozione
        # split('_', 1) divide solo al primo underscore. "185_anger" diventa ["185", "anger"]
        parts = name_without_ext.split('_', 1)
        
        # Se il file non ha l'underscore, lo saltiamo (non rispetta il formato)
        if len(parts) < 2:
            continue
            
        # Prendiamo la seconda parte e la mettiamo in minuscolo per sicurezza
        emotion_str = parts[1].lower()
        
        # Controllo VIP: L'emozione fa parte del nostro progetto?
        if emotion_str not in EMOTION_MAP:
            # Se è "baseline", "all" o "neutral", lo ignoriamo bellamente
            continue
            
        label = EMOTION_MAP[emotion_str]
        
        # ==========================================
        # 3. LETTURA DEI METADATI DAL FILE
        # ==========================================
        subject_id = None
        total_lines = 0
        
        # Apriamo il file in sola lettura per contare le righe e prendere il soggetto
        # Nota: NON usiamo pandas qui per leggere tutto, sennò la RAM esplode!
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # La riga 2 (indice 1) contiene il Subject_ID (es: #Subject_ID,185)
                if i == 1:
                    subject_id_parts = line.strip().split(',')
                    if len(subject_id_parts) > 1:
                        subject_id = subject_id_parts[1] # Prende il "185"
                
                # Contiamo quante righe ci sono in totale
                total_lines += 1
                
        # Le prime 9 righe sono l'intestazione, quindi le righe di dati veri sono:
        data_rows = total_lines - 9
        
        # Se il file è vuoto o troppo corto, lo saltiamo
        if data_rows < WINDOW_SIZE:
            continue
            
        # ==========================================
        # 4. CREAZIONE DELLE FINESTRE (WINDOWING)
        # ==========================================
        # Ora sappiamo quanti dati ci sono. Li "affettiamo" in blocchi da 1000.
        # Es. Se data_rows è 5500, creeremo 5 finestre complete.
        num_windows = data_rows // WINDOW_SIZE
        
        # La riga 10 (indice 9) è dove iniziano i numeri veri nel file CSV
        data_start_index = 9 
        
        for w in range(num_windows):
            # --- MAGIA DELL'UNDERSAMPLING SULLE FINESTRE ---
            # Se l'emozione è POSITIVA (1), teniamo solo 1 finestra su 3 (w % 3 == 0)
            # Questo dimezza drasticamente i positivi, bilanciando il dataset!
            if label == 1 and w % 3 != 0:
                continue # Salta questa fetta e passa alla prossima
            # -----------------------------------------------

            # Calcoliamo la riga esatta di partenza e di fine per questa specifica fettina
            start_row = data_start_index + (w * WINDOW_SIZE)
            end_row = start_row + WINDOW_SIZE
            
            # Salviamo le coordinate di questa fettina nella nostra lista
            index_data.append({
                'file_path': filepath,
                'subject_id': subject_id,
                'emotion': emotion_str,
                'label': label,
                'start_row': start_row,
                'end_row': end_row
            })

    # ==========================================
    # 5. SALVATAGGIO DELL'INDICE
    # ==========================================
    # Convertiamo la lista di dizionari in un comodo DataFrame Pandas
    df_index = pd.DataFrame(index_data)
    
    # Creiamo la cartella processed se non esiste
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Salviamo l'indice in un nuovo file CSV
    output_path = os.path.join(PROCESSED_DIR, "dataset_index.csv")
    df_index.to_csv(output_path, index=False)
    
    print(f"Finito! Indice creato con {len(df_index)} 'fettine' (esempi di addestramento).")
    print(f"File salvato in: {output_path}")

# Se eseguiamo questo script direttamente, lancia la funzione
if __name__ == "__main__":
    create_dataset_index()