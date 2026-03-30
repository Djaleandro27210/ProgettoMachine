"""
Training module for Phase 1 (ECG-only model).

This module implements the complete training loop for the 1D-CNN model:
- Data loading (train/val splits)
- Forward pass with loss computation
- Backpropagation and weight updates (Adam optimizer)
- Learning rate scheduling
- Model checkpointing based on validation F1-score
"""

import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config_FASE1 import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    WEIGHT_DEECAY,
)
from dataset_FASE1 import PopaneDataset
from model_FASE1 import Emotion1DCNN


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Training hyperparameters
CLASSIFICATION_THRESHOLD = 0.5
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 4
LR_SCHEDULER_FACTOR = 0.5


def get_device() -> torch.device:
    """Determine computing device (GPU or CPU).
    
    Returns:
        torch.device: CUDA if available, otherwise CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def load_datasets(
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """Load train and validation datasets.
    
    Args:
        batch_size: Batch size for DataLoaders.
        
    Returns:
        Tuple of (train_loader, val_loader).
    """
    logger.info("Loading train and validation datasets...")
    train_dataset = PopaneDataset(split_type="train")
    val_dataset = PopaneDataset(split_type="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader


def compute_class_weights(train_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Compute class weights for imbalanced dataset.
    
    Args:
        train_loader: DataLoader for training set.
        device: Computing device.
        
    Returns:
        pos_weight tensor for BCEWithLogitsLoss.
    """
    logger.info("Computing class weights for balance...")
    
    num_positives = 0
    num_negatives = 0
    
    for _, labels in train_loader:
        num_positives += labels.sum().item()
        num_negatives += (labels == 0).sum().item()
    
    total_samples = num_positives + num_negatives
    weight_value = num_negatives / num_positives if num_positives > 0 else 1.0
    
    logger.info(f"Class distribution: Negative={int(num_negatives)} ({100*num_negatives/total_samples:.1f}%), "
                f"Positive={int(num_positives)} ({100*num_positives/total_samples:.1f}%)")
    logger.info(f"Computed pos_weight: {weight_value:.2f}")
    
    return torch.tensor([weight_value]).to(device)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Execute a single training epoch.
    
    Args:
        model: Neural network model.
        train_loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Computing device.
        
    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        predictions = (torch.sigmoid(outputs) > CLASSIFICATION_THRESHOLD).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 500 == 0:
            logger.debug(f"Batch {batch_idx + 1}/{len(train_loader)}")
    
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run validation on the validation set.
    
    Args:
        model: Neural network model.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Computing device.
        
    Returns:
        Dictionary with loss, accuracy, and F1-score.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            predictions = (torch.sigmoid(outputs) > CLASSIFICATION_THRESHOLD).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total if total > 0 else float("inf")
    accuracy = (correct / total * 100) if total > 0 else 0.0
    f1_macro = f1_score(all_labels, all_preds, average="macro") if len(set(all_labels)) > 1 else 0.0
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_score": f1_macro,
    }


def train_model(
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    model_save_path: str = MODEL_SAVE_PATH,
) -> None:
    """Main training loop.
    
    Args:
        epochs: Number of training epochs.
        learning_rate: Initial learning rate.
        batch_size: Batch size for training.
        model_save_path: Path to save the best model.
    """
    device = get_device()
    
    # Load data
    train_loader, val_loader = load_datasets(batch_size=batch_size)
    
    # Initialize model
    model = Emotion1DCNN().to(device)
    
    # Compute class weights
    pos_weight = compute_class_weights(train_loader, device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DEECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
    )
    
    # Early stopping and checkpoint tracking
    best_val_f1 = 0.0
    epochs_without_improvement = 0
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Log epoch results
        logger.info(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
            f"Val: Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.2f}%, "
            f"F1={val_metrics['f1_score']:.4f}"
        )
        
        # Learning rate scheduling
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr < learning_rate:
            logger.info(f"Learning rate reduced to: {current_lr:.6f}")
        
        # Checkpoint logic based on F1-score
        if val_metrics["f1_score"] > best_val_f1:
            best_val_f1 = val_metrics["f1_score"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"New best F1-score: {best_val_f1:.4f}! Model saved.")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epochs")
        
        # Early stopping
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    logger.info(f"Training completed. Best Val F1-score: {best_val_f1:.4f}")


if __name__ == "__main__":
    train_model()