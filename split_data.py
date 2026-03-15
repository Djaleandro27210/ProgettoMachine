import pandas as pd
import numpy as np

# ==========================================
# IMPORTIAMO I PERCORSI DAL CONFIG
# ==========================================
from config import INDEX_PATH, SPLIT_INDEX_PATH

def perform_subject_split():
    # Imposta un seed fisso per la riproducibilità scientifica!
    np.random.seed(42)

    # 1. Carica l'indice che hai appena creato usando la costante dal config
    print(f"Caricamento di {INDEX_PATH} in corso...")
    df = pd.read_csv(INDEX_PATH)

    # 2. Trova tutti i soggetti UNICI (es. 100, 101, 185, ecc.)
    unique_subjects = df['subject_id'].unique()
    print(f"Trovati {len(unique_subjects)} soggetti unici in totale.")

    # 3. Mescola l'ordine dei soggetti casualmente
    np.random.shuffle(unique_subjects)

    # 4. Calcola le quote per lo split (70% Train, 15% Val, 15% Test)
    total_subjects = len(unique_subjects)
    n_train = int(total_subjects * 0.70)
    n_val = int(total_subjects * 0.15)

    # Dividi i soggetti nei tre gruppi
    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:n_train + n_val]
    test_subjects = unique_subjects[n_train + n_val:]

    print(f"Divisione Soggetti -> Train: {len(train_subjects)} | Val: {len(val_subjects)} | Test: {len(test_subjects)}")

    # 5. Funzione che controlla a quale gruppo appartiene il soggetto
    def assign_split(subject_id):
        if subject_id in train_subjects:
            return 'train'
        elif subject_id in val_subjects:
            return 'val'
        else:
            return 'test'

    # 6. Applica la funzione a tutte le righe per creare la colonna 'split'
    print("Assegnazione delle fettine ai vari set in corso...")
    df['split'] = df['subject_id'].apply(assign_split)

    # 7. Salva il nuovo file usando la costante dal config
    df.to_csv(SPLIT_INDEX_PATH, index=False)
    
    print(f"\nFinito! Il nuovo indice con lo split è stato salvato in: {SPLIT_INDEX_PATH}")
    
    # Stampa quante "fettine" effettive sono finite in ogni set
    print("\nRiepilogo finale delle FETTINE per ogni set:")
    print(df['split'].value_counts())

if __name__ == "__main__":
    perform_subject_split()