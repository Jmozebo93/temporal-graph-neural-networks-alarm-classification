from __future__ import annotations

# Metrics and plotting utilities for model evaluation.
# This module computes performance metrics and generates the diagnostic charts
# used to assess the Temporal GNN after training.

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import auc, classification_report, confusion_matrix, f1_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_model(model: torch.nn.Module, loader, device: torch.device, class_weights: torch.Tensor) -> Tuple[float, float, Dict[str, object]]:
    # Evaluate a trained model over the provided loader and return loss plus
    # standard classification metrics for the active window-level predictions.
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.y.view(-1))
            total_loss += float(loss.item())

            preds = logits.argmax(dim=1).detach().cpu().tolist()
            probs = torch.softmax(logits, dim=1).detach().cpu().tolist()
            labels = batch.y.view(-1).detach().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)

    accuracy = float(np.mean(np.asarray(all_preds) == np.asarray(all_labels))) if all_labels else 0.0
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0)) if all_labels else 0.0
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0) if all_labels else {}
    matrix = confusion_matrix(all_labels, all_preds).tolist() if all_labels else []

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": matrix,
        "labels": all_labels,
        "predictions": all_preds,
        "probabilities": all_probs,
    }
    return accuracy, total_loss / max(len(loader), 1), metrics


def save_confusion_matrix_plot(conf_matrix: List[List[int]], class_names: List[str], output_path: str) -> None:
    # Save a confusion matrix heatmap showing the true and predicted classes.
    matrix = np.asarray(conf_matrix)
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if matrix.size > 0:
        threshold = matrix.max() / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = int(matrix[i, j])
                color = "white" if value > threshold else "black"
                plt.text(j, i, str(value), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_loss_curve(train_losses: List[float], val_losses: List[float], output_path: str) -> None:
    # Plot train and validation loss over time to inspect convergence behavior.
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_multiclass_roc_curve(labels: List[int], probabilities: List[List[float]], class_names: List[str], output_path: str) -> Dict[str, float]:
    # Generate one-vs-rest ROC curves for each class and save them to disk.
    y_true = np.asarray(labels)
    y_prob = np.asarray(probabilities)
    roc_auc: Dict[str, float] = {}

    plt.figure(figsize=(8, 6))
    for idx, class_name in enumerate(class_names):
        binary_true = (y_true == idx).astype(int)
        if len(np.unique(binary_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(binary_true, y_prob[:, idx])
        class_auc = float(auc(fpr, tpr))
        roc_auc[class_name] = class_auc
        plt.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC={class_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return roc_auc
