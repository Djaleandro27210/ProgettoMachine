"""
=====================================================================================
Module: train_FASE2_early.py
Project: ML Emotions - Phase 2 (Early Fusion Training)

Description:
Training script for the 3-channel Early Fusion architecture.
Integrated with Early Stopping and Learning Rate Scheduler.
Saves the best model on disk BASED ON MACRO F1-SCORE.
=====================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score # <-- IMPORTANT IMPORT FOR F1

# ⚠️ WARNING: Importing from Phase 2 files!
from config_FASE2_early import BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_SAVE_PATH_FASE2
from dataset_FASE2_early import PopaneDatasetMultimodal
from model_FASE2_early import MultimodalEarlyFusionCNN


def get_device() -> torch.device:
    """Returns the available device for training."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_data_loaders(batch_size: int = BATCH_SIZE):
    """Builds the DataLoader for training and validation."""
    train_dataset = PopaneDatasetMultimodal(split_type="train")
    val_dataset = PopaneDatasetMultimodal(split_type="val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader


def build_model(device: torch.device):
    """Creates the network model and sends it to the device."""
    model = MultimodalEarlyFusionCNN()
    return model.to(device)


def build_optimizer(model: torch.nn.Module):
    """Creates the optimizer."""
    return optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)


def build_scheduler(optimizer):
    """Creates the LR scheduler."""
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)


def compute_pos_weight(train_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Calculates the weight for the positive class for BCEWithLogitsLoss."""
    positives = 0
    negatives = 0

    for _, labels in train_loader:
        positives += int(labels.sum().item())
        negatives += int((labels == 0).sum().item())

    if positives <= 0:
        return torch.tensor([1.0], dtype=torch.float32, device=device)

    weight_value = negatives / positives
    return torch.tensor([weight_value], dtype=torch.float32, device=device)


def batch_metrics(outputs: torch.Tensor, labels: torch.Tensor):
    """Binary predictions and basic batch metrics."""
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()
    loss_labels = labels.view_as(preds)
    correct = (preds == loss_labels).sum().item()
    total = loss_labels.numel()

    return preds, loss_labels, correct, total


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (inputs, labels) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        labels = labels.to(device).unsqueeze(1).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds, true_labels, batch_correct, batch_total = batch_metrics(outputs, labels)

        total_loss += float(loss.item()) * batch_total
        correct += batch_correct
        total += batch_total

        if batch_idx % 500 == 0:
            print(f"   -> Processed batch {batch_idx}/{len(loader)}")

    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def evaluate(model: torch.nn.Module, loader: DataLoader, criterion, device: torch.device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            preds, true_labels, batch_correct, batch_total = batch_metrics(outputs, labels)

            total_loss += float(loss.item()) * batch_total
            correct += batch_correct
            total += batch_total

            all_preds.extend(preds.cpu().numpy().ravel().tolist())
            all_labels.extend(true_labels.cpu().numpy().ravel().tolist())

    if total == 0:
        return float('inf'), 0.0, 0.0

    val_loss = total_loss / total
    val_acc = 100.0 * correct / total
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return val_loss, val_acc, val_f1

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            preds, true_labels, batch_correct, batch_total = batch_metrics(outputs, labels)

            total_loss += float(loss.item()) * batch_total
            correct += batch_correct
            total += batch_total

            all_preds.extend(preds.cpu().numpy().ravel().tolist())
            all_labels.extend(true_labels.cpu().numpy().ravel().tolist())

    if total == 0:
        return float('inf'), 0.0, 0.0

    val_loss = total_loss / total
    val_acc = 100.0 * correct / total
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return val_loss, val_acc, val_f1


def train_early_fusion():
    device = get_device()
    print(f"Starting EARLY FUSION training. Device: {device}")

    train_loader, val_loader = create_data_loaders()
    model = build_model(device)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    pos_weight = compute_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Calculated pos_weight: {pos_weight.item():.4f}")

    patience_early_stopping = 10
    epochs_no_improve = 0
    best_val_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch}/{EPOCHS}] - Train Loss: {train_loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% - Val F1-Macro: {val_f1:.4f}")

        scheduler.step(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH_FASE2)
            print(f"New F1-Macro record {best_val_f1:.4f}, model saved in {MODEL_SAVE_PATH_FASE2}.")
        else:
            epochs_no_improve += 1
            print(f"No F1 improvement for {epochs_no_improve}/{patience_early_stopping} epochs.")

        if epochs_no_improve >= patience_early_stopping:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"Training completed. Best Val F1-Macro: {best_val_f1:.4f}")


if __name__ == "__main__":
    train_early_fusion()