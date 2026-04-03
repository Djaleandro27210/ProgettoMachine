"""
train_FASE2_early.py
Training loop for the Early Fusion 3-channel architecture.
Implements dynamic class weighting, learning rate scheduling, and F1-based early stopping.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from config_FASE2_early import BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_SAVE_PATH_FASE2, WEIGHT_DECAY
from dataset_FASE2_early import PopaneDatasetMultimodal
from model_FASE2_early import MultimodalEarlyFusionCNN


def calculate_class_weights(train_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Computes the positive class weight to offset dataset imbalance."""
    num_positives = 0
    num_negatives = 0
    
    for inputs, labels in train_loader:
        num_positives += labels.sum().item()
        num_negatives += (labels == 0).sum().item()

    weight_value = num_negatives / num_positives if num_positives > 0 else 1.0
    
    print(f"[INFO] Class distribution - Negative: {int(num_negatives)} | Positive: {int(num_positives)}")
    print(f"[INFO] Computed pos_weight multiplier: {weight_value:.4f}")
    
    return torch.tensor([weight_value]).to(device)


def train_early_fusion() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing Phase 2 (Early Fusion) Training on: {device}")

    print("[INFO] Loading multimodal datasets...")
    train_dataset = PopaneDatasetMultimodal(split_type="train")
    val_dataset = PopaneDatasetMultimodal(split_type="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = MultimodalEarlyFusionCNN().to(device)
    pos_weight = calculate_class_weights(train_loader, device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=4
    )

    patience_early_stopping = 10
    epochs_no_improve = 0
    best_val_f1 = 0.0 
    prev_lr = LEARNING_RATE

    print(f"[INFO] Beginning training loop for {EPOCHS} epochs...\n")
    
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        running_train_loss = 0.0
        total_train = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            total_train += labels.size(0)

            if (batch_idx + 1) % 500 == 0:
                print(f"    -> Processed batch {batch_idx + 1}/{len(train_loader)}")

        epoch_train_loss = running_train_loss / total_train

        # --- Validation ---
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct_val += (predictions == labels).sum().item()
                total_val += labels.size(0)
                
                all_val_preds.extend(predictions.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        if total_val > 0:
            epoch_val_loss = running_val_loss / total_val
            epoch_val_acc = (correct_val / total_val) * 100
            epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        else:
            epoch_val_loss = float('inf')
            epoch_val_acc = 0.0
            epoch_val_f1 = 0.0
            print("[WARNING] Validation set is empty.")

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.2f}% - Val F1-Macro: {epoch_val_f1:.4f}")

        # --- Scheduling and Checkpointing ---
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        if current_lr < prev_lr: 
            print(f"[INFO] Learning rate reduced to: {current_lr}")
            prev_lr = current_lr

        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            epochs_no_improve = 0 
            torch.save(model.state_dict(), MODEL_SAVE_PATH_FASE2)
            print(f"[INFO] New best F1-Macro ({best_val_f1:.4f}). Model saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience_early_stopping:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch+1}.")
            break 

    print(f"\n[INFO] Training complete. Best Validation F1-Macro: {best_val_f1:.4f}")


if __name__ == "__main__":
    train_early_fusion()