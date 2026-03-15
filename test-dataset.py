from dataset import PopaneDataset
from torch.utils.data import DataLoader
from config import BATCH_SIZE, WINDOW_SIZE

def test_data_pipeline():
    print("Inizializzazione Dataset...")
    # Guarda qui: non gli passiamo più l'index_path! Fa tutto da solo.
    train_dataset = PopaneDataset(split_type="train")
    
    print(f"Dimensione del Training Set: {len(train_dataset)} esempi")
    
    # Estraiamo manualmente il primissimo esempio (indice 0)
    x, y = train_dataset[0]
    
    print("\n--- TEST PRIMA FETTINA ---")
    print(f"Forma del Tensore X (ECG): {x.shape}")
    print(f"Valore del Tensore Y (Label): {y.item()}")
    
    # Inizializziamo il DataLoader usando il BATCH_SIZE dal config
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Proviamo a pescare il primo batch
    batch_x, batch_y = next(iter(train_loader))
    print("\n--- TEST PRIMO BATCH ---")
    print(f"Forma del Batch X: {batch_x.shape} -> (Batch_Size={BATCH_SIZE}, Canali=1, Lunghezza={WINDOW_SIZE})")
    print(f"Forma del Batch Y: {batch_y.shape}")
    print("\nSe i numeri coincidono, il motore funziona da dio! Sei pronto per la rete neurale.")

if __name__ == "__main__":
    test_data_pipeline()