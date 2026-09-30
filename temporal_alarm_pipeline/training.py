from __future__ import annotations

# Training utilities for the Temporal GNN pipeline.
# This module includes data loaders, balancing helpers, optimization logic, and
# the core training loop used by the active training entrypoint.

import random
from collections import Counter
from typing import Dict, List, Sequence

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler


def make_loader(graphs: Sequence[Data], batch_size: int, shuffle: bool) -> DataLoader:
    # Wrap graph windows in a PyG data loader for batching during training or eval.
    return DataLoader(list(graphs), batch_size=batch_size, shuffle=shuffle)


def make_balanced_loader(graphs: Sequence[Data], batch_size: int) -> DataLoader:
    # Build a class-balanced loader for experiments that need equal label exposure.
    graphs = list(graphs)
    label_counts = Counter(int(graph.y.item()) for graph in graphs)
    sample_weights = [1.0 / label_counts[int(graph.y.item())] for graph in graphs]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(graphs, batch_size=batch_size, sampler=sampler)


def rebalance_graphs(graphs: Sequence[Data], target_per_class: int, seed: int = 21) -> List[Data]:
    # Rebalance window graphs across classes so minority classes are not ignored.
    rng = random.Random(seed)
    buckets: Dict[int, List[Data]] = {}
    for graph in graphs:
        buckets.setdefault(int(graph.y.item()), []).append(graph)

    balanced: List[Data] = []
    for label in sorted(buckets):
        bucket = buckets[label]
        if len(bucket) >= target_per_class:
            balanced.extend(rng.sample(bucket, target_per_class))
        else:
            balanced.extend(bucket)
            balanced.extend(rng.choices(bucket, k=target_per_class - len(bucket)))

    rng.shuffle(balanced)
    return balanced


def compute_class_weights(graphs: Sequence[Data], num_classes: int) -> torch.Tensor:
    # Compute inverse-frequency class weights to reduce bias toward majority labels.
    counts = Counter(int(graph.y.item()) for graph in graphs)
    total = sum(counts.values())
    weights = []
    for idx in range(num_classes):
        count = counts.get(idx, 0)
        weights.append(0.0 if count == 0 else total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, class_weights: torch.Tensor) -> float:
    # Run a full training epoch with weighted cross-entropy loss.
    model.train()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())

    return total_loss / max(len(loader), 1)


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    class_weights,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int = 12,
    min_delta: float = 1e-4,
):
    # Train the model while monitoring training/validation accuracy and macro-F1,
    # then save the best checkpoint encountered during the run.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_f1 = -1.0
    best_val_accuracy = -1.0
    best_path = "temporal_alarm_tgnn_best_model.pt"
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, class_weights)
        train_acc, _, _ = evaluate_model(model, train_loader, device, class_weights)
        val_acc, val_loss, val_metrics = evaluate_model(model, val_loader, device, class_weights)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        scheduler.step(val_metrics["macro_f1"])

        if val_metrics["macro_f1"] > best_val_f1 + min_delta:
            best_val_f1 = val_metrics["macro_f1"]
            best_val_accuracy = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        if epoch % 1 == 0:
            print(
                f"Epoch {epoch:03d} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
                f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | Val F1 {val_metrics['macro_f1']:.4f}"
            )

        if patience_counter >= early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}: validation macro-F1 has not improved by at least "
                f"{min_delta:.4f} for {early_stopping_patience} consecutive epochs."
            )
            break

    final_train_acc, _, final_train_metrics = evaluate_model(model, train_loader, device, class_weights)
    final_train_report = final_train_metrics.get("classification_report", {})
    final_train_macro = final_train_report.get("macro avg", {})

    return {
        "best_model_path": best_path,
        "best_val_f1": best_val_f1,
        "best_val_accuracy": best_val_accuracy,
        "final_train_accuracy": final_train_acc if final_train_acc else train_accuracies[-1] if train_accuracies else 0.0,
        "final_train_precision": float(final_train_macro.get("precision", 0.0)),
        "final_train_recall": float(final_train_macro.get("recall", 0.0)),
        "final_train_f1": float(final_train_macro.get("f1-score", 0.0)),
        "final_val_accuracy": val_accuracies[-1] if val_accuracies else 0.0,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "final_train_classification_report": final_train_report,
        "optimizer": optimizer,
    }


def evaluate_model(model, loader, device, class_weights):
    # Delegate to the evaluation module so the training loop and evaluation script
    # share the same metric implementation.
    try:
        from .evaluation import evaluate_model as evaluate
    except ImportError:  # pragma: no cover - fallback for direct script execution
        from evaluation import evaluate_model as evaluate

    return evaluate(model, loader, device, class_weights)
