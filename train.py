from __future__ import annotations

# Top-level training entrypoint for the active Temporal GNN workflow.
# This script orchestrates data loading, graph construction, training,
# checkpointing, and artifact generation for the production path.

import argparse
import json
from collections import Counter

import numpy as np
import torch

from temporal_alarm_pipeline.data import TemporalGraphBuilder, load_csv_records
from temporal_alarm_pipeline.evaluation import (
    save_confusion_matrix_plot,
    save_loss_curve,
    save_multiclass_roc_curve,
)
from temporal_alarm_pipeline.model import TemporalEventGNN
from temporal_alarm_pipeline.training import compute_class_weights, make_loader, rebalance_graphs, train_model


def build_parser() -> argparse.ArgumentParser:
    # Define the command-line interface for the training workflow.
    parser = argparse.ArgumentParser(description="Train the temporal alarm GNN")
    parser.add_argument("--train-csv", default="UUI_synthetic_data_generator/balanced_synthetic_data_train.csv")
    parser.add_argument("--validation-csv", default="UUI_synthetic_data_generator/balanced_synthetic_data_validation.csv")
    parser.add_argument("--label-field", default="ground_truth", choices=["ground_truth", "assessment"])
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--output-prefix", default="temporal_alarm_tgnn")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    # Main training orchestration: load data, build graphs, train the model,
    # and save the best checkpoint plus diagnostic plots.
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 72)
    print("TRAIN TEMPORAL ALARM T-GNN")
    print("=" * 72)
    print(f"Label field: {args.label_field}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Device: {args.device}")
    print()

    # Load raw records and convert them into temporal graph instances.
    train_records = load_csv_records(args.train_csv)
    validation_records = load_csv_records(args.validation_csv)
    builder = TemporalGraphBuilder(
        label_field=args.label_field,
        window_size=args.window_size,
        stride=args.stride,
    )

    train_labels = {builder.label_fn(record) for record in train_records}
    needs_training_merge = any(label not in train_labels for label in builder.label_names)

    if needs_training_merge:
        # Some classes may be missing in the train split; merge the splits so
        # training can still build a complete supervised classifier.
        print("Train split does not contain every class; combining train + validation for supervised training.")
        merged_records = train_records + validation_records
        merged_graphs = builder.build_graphs(merged_records, fit_vectorizer=True)
        split_at = max(1, int(round(len(merged_graphs) * 0.85)))
        train_graphs = merged_graphs[:split_at]
        val_graphs = merged_graphs[split_at:]
    else:
        print("Building train graphs...")
        train_graphs = builder.build_graphs(train_records, fit_vectorizer=True)
        print("Building validation graphs...")
        val_graphs = builder.build_graphs(validation_records)

    if args.label_field == "ground_truth":
        # Keep class frequencies balanced for the three-class problem.
        train_counts = Counter(g.label_text for g in train_graphs)
        target_per_class = min(500, max(train_counts.values()))
        print(f"Rebalancing train graphs to {target_per_class} per class for 3-class training.")
        train_graphs = rebalance_graphs(train_graphs, target_per_class=target_per_class, seed=args.seed)

    if not train_graphs or not val_graphs:
        raise RuntimeError("Training or validation graph set is empty. Reduce window-size or inspect the CSVs.")

    # Instantiate the model and prepare data loaders for training.
    device = torch.device(args.device)
    model = TemporalEventGNN(
        input_dim=train_graphs[0].x.size(1),
        hidden_dim=args.hidden_dim,
        num_classes=len(builder.label_names),
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    train_loader = make_loader(train_graphs, args.batch_size, shuffle=True)
    val_loader = make_loader(val_graphs, args.batch_size, shuffle=False)
    class_weights = compute_class_weights(train_graphs, len(builder.label_names))

    summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=class_weights,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
    )

    # Load the best checkpoint before generating final validation plots.
    best_path = summary["best_model_path"]
    model.load_state_dict(torch.load(best_path, map_location=device))

    val_loader = make_loader(val_graphs, args.batch_size, shuffle=False)
    _, _, val_metrics = __import__('temporal_alarm_pipeline.evaluation', fromlist=['evaluate_model']).evaluate_model(
        model, val_loader, device, class_weights
    )

    # Save the standard diagnostic artifacts used to report model quality.
    cm_plot_path = f"{args.output_prefix}_confusion_matrix.png"
    loss_plot_path = f"{args.output_prefix}_loss_curve.png"
    roc_plot_path = f"{args.output_prefix}_roc_curve.png"
    save_confusion_matrix_plot(val_metrics["confusion_matrix"], builder.label_names, cm_plot_path)
    save_loss_curve(summary["train_losses"], summary["val_losses"], loss_plot_path)
    roc_auc = save_multiclass_roc_curve(
        labels=val_metrics["labels"],
        probabilities=val_metrics["probabilities"],
        class_names=builder.label_names,
        output_path=roc_plot_path,
    )

    training_summary_path = f"{args.output_prefix}_training_summary.json"
    training_summary = {
        "best_model_path": summary["best_model_path"],
        "best_val_f1": float(summary["best_val_f1"]),
        "best_val_accuracy": float(summary["best_val_accuracy"]),
        "final_train_accuracy": float(summary["final_train_accuracy"]),
        "final_train_precision": float(summary["final_train_precision"]),
        "final_train_recall": float(summary["final_train_recall"]),
        "final_train_f1": float(summary["final_train_f1"]),
        "final_val_accuracy": float(summary["final_val_accuracy"]),
        "train_losses": summary["train_losses"],
        "val_losses": summary["val_losses"],
        "train_accuracies": summary["train_accuracies"],
        "val_accuracies": summary["val_accuracies"],
        "final_train_classification_report": summary["final_train_classification_report"],
    }
    with open(training_summary_path, "w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2)

    print()
    print(f"Saved best model to {best_path}")
    print(f"Best validation macro F1: {summary['best_val_f1']:.4f}")
    print(f"Best validation accuracy: {summary['best_val_accuracy']:.4f}")
    print(f"Final training accuracy: {summary['final_train_accuracy']:.4f}")
    print(f"Final training precision: {summary['final_train_precision']:.4f}")
    print(f"Final training recall: {summary['final_train_recall']:.4f}")
    print(f"Final training F1: {summary['final_train_f1']:.4f}")
    print(f"Final validation accuracy: {summary['final_val_accuracy']:.4f}")
    print(f"Saved training summary to {training_summary_path}")
    print(f"Saved confusion matrix plot to {cm_plot_path}")
    print(f"Saved loss curve plot to {loss_plot_path}")
    print(f"Saved ROC curve plot to {roc_plot_path}")
    print(f"Validation ROC AUC: {roc_auc}")


if __name__ == "__main__":
    main()
