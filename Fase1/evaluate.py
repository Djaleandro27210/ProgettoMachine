"""
Evaluation module for Phase 1 (ECG-only model).

This script performs inference on the test set and generates comprehensive
evaluation metrics including accuracy, F1-score, AUC-ROC, classification report,
and confusion matrix. Results are printed to console and saved to output_fase1.txt.
"""

import logging
import os
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from config_FASE1 import BATCH_SIZE, MODEL_SAVE_PATH
from dataset_FASE1 import PopaneDataset
from model_FASE1 import Emotion1DCNN


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASSIFICATION_THRESHOLD = 0.5
OUTPUT_FILENAME = "output_fase1.txt"
CLASS_NAMES = ["Negative Emotion (0)", "Positive Emotion (1)"]


def get_device() -> torch.device:
    """Determine the appropriate computing device (GPU or CPU).
    
    Returns:
        torch.device: CUDA if available, otherwise CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def load_model(model_path: str, device: torch.device) -> Emotion1DCNN:
    """Load pre-trained model from disk.
    
    Args:
        model_path: Path to saved model weights.
        device: Target device (CPU or GPU).
        
    Returns:
        Model in evaluation mode.
        
    Raises:
        RuntimeError: If model cannot be loaded.
    """
    model = Emotion1DCNN().to(device)
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        logger.info(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    model.eval()
    return model


def run_inference(
    model: Emotion1DCNN,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference on test set and collect predictions.
    
    Args:
        model: Trained model in eval mode.
        test_loader: DataLoader for test set.
        device: Computing device.
        
    Returns:
        Tuple of (true_labels, predictions, probabilities) as numpy arrays.
    """
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    logger.info("Starting inference on test set...")
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            logger.info(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > CLASSIFICATION_THRESHOLD).float()
            
            # Handle single-element batches
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
                preds = preds.unsqueeze(0)
            
            all_probabilities.extend(probs.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    logger.info("Inference completed")
    
    return (
        np.array(all_labels).astype(int),
        np.array(all_predictions).astype(int),
        np.array(all_probabilities),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, any]:
    """Compute evaluation metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels (0 or 1).
        y_prob: Predicted probabilities.
        
    Returns:
        Dictionary containing all computed metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        logger.warning("Could not compute AUC-ROC (single class in test set)")
        auc_roc = float("nan")
    
    class_report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1-Score (Macro): {f1_macro:.4f}")
    logger.info(f"AUC-ROC: {auc_roc:.4f}")
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "auc_roc": auc_roc,
        "class_report": class_report,
        "confusion_matrix": conf_matrix,
    }


def format_report(metrics: Dict[str, any]) -> str:
    """Format metrics into a readable report string.
    
    Args:
        metrics: Dictionary from compute_metrics().
        
    Returns:
        Formatted report string.
    """
    cm = metrics["confusion_matrix"]
    
    report_text = f"""==================================================
EVALUATION RESULTS - PHASE 1 (ECG)
==================================================
Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)
F1-Score:  {metrics['f1_macro']:.4f} (Macro Average)
AUC-ROC:   {metrics['auc_roc']:.4f}

--------------------------------------------------
CLASSIFICATION REPORT
--------------------------------------------------
{metrics['class_report']}
--------------------------------------------------
CONFUSION MATRIX
--------------------------------------------------
True Negative (TN):   {cm[0][0]:<5} | False Positive (FP): {cm[0][1]}
False Negative (FN):  {cm[1][0]:<5} | True Positive (TP):  {cm[1][1]}
==================================================
"""
    return report_text


def save_report(report_text: str, output_filename: str = OUTPUT_FILENAME) -> None:
    """Save evaluation report to file.
    
    Args:
        report_text: Formatted report string.
        output_filename: Name of output file (saved in script directory).
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(current_dir, output_filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {output_path}")
    except IOError as e:
        logger.error(f"Failed to save report: {e}")


def evaluate_model() -> None:
    """Main evaluation pipeline."""
    device = get_device()
    
    # Load test set
    logger.info("Loading test set...")
    test_dataset = PopaneDataset(split_type="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    model = load_model(MODEL_SAVE_PATH, device)
    
    # Run inference
    y_true, y_pred, y_prob = run_inference(model, test_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    # Format and display report
    report_text = format_report(metrics)
    print(report_text)
    
    # Save report
    save_report(report_text)


if __name__ == "__main__":
    evaluate_model()