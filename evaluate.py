from __future__ import annotations

# Top-level evaluation entrypoint for the active Temporal GNN workflow.
# This script loads a trained model, evaluates it on held-out data, and saves
# the metrics and plots used to assess production readiness.

import argparse
import json

import numpy as np
import torch

from temporal_alarm_pipeline.data import TemporalGraphBuilder, load_csv_records
from temporal_alarm_pipeline.evaluation import (
    evaluate_model,
    save_confusion_matrix_plot,
    save_multiclass_roc_curve,
)
from temporal_alarm_pipeline.model import TemporalEventGNN
from temporal_alarm_pipeline.training import compute_class_weights, make_loader


def build_parser() -> argparse.ArgumentParser:
    # Define the command-line interface for evaluation.
    parser = argparse.ArgumentParser(description="Evaluate the trained temporal alarm GNN")
    parser.add_argument("--train-csv", default="UUI_synthetic_data_generator/balanced_synthetic_data_train.csv")
    parser.add_argument("--test-csv", default="UUI_synthetic_data_generator/balanced_synthetic_data_scenario.csv")
    parser.add_argument("--label-field", default="ground_truth", choices=["ground_truth", "assessment"])
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-path", default="temporal_alarm_tgnn_best_model.pt")
    parser.add_argument("--output-prefix", default="temporal_alarm_tgnn")
    parser.add_argument("--seed", type=int, default=21)
    return parser


def main() -> None:
    # Main evaluation orchestration: build test graphs, load the model,
    # compute metrics, and persist the summary outputs.
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 72)
    print("EVALUATE TEMPORAL ALARM T-GNN")
    print("=" * 72)
    print(f"Model: {args.model_path}")
    print(f"Label field: {args.label_field}")
    print()

    # Fit the vectorizer on the training data and then build test graphs from the
    # held-out dataset using the same feature space.
    train_records = load_csv_records(args.train_csv)
    test_records = load_csv_records(args.test_csv)
    builder = TemporalGraphBuilder(
        label_field=args.label_field,
        window_size=args.window_size,
        stride=args.stride,
    )
    builder.build_graphs(train_records, fit_vectorizer=True)
    test_graphs = builder.build_graphs(test_records)

    if not test_graphs:
        # A missing test graph set usually means the window settings are too large
        # or the dataset is not compatible with the configured windowing.
        raise RuntimeError("No test graphs were generated. Reduce window-size or inspect the CSVs.")

    # Load the trained checkpoint and run it in evaluation mode.
    device = torch.device(args.device)
    model = TemporalEventGNN(
        input_dim=test_graphs[0].x.size(1),
        hidden_dim=args.hidden_dim,
        num_classes=len(builder.label_names),
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    test_loader = make_loader(test_graphs, args.batch_size, shuffle=False)
    class_weights = compute_class_weights(test_graphs, len(builder.label_names))
    test_acc, test_loss, test_metrics = evaluate_model(model, test_loader, device, class_weights)

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print("Confusion Matrix:")
    print(np.asarray(test_metrics["confusion_matrix"]))

    # Save plot artifacts and a machine-readable summary of model performance.
    cm_plot_path = f"{args.output_prefix}_confusion_matrix.png"
    roc_plot_path = f"{args.output_prefix}_roc_curve.png"
    save_confusion_matrix_plot(test_metrics["confusion_matrix"], builder.label_names, cm_plot_path)
    roc_auc = save_multiclass_roc_curve(
        labels=test_metrics["labels"],
        probabilities=test_metrics["probabilities"],
        class_names=builder.label_names,
        output_path=roc_plot_path,
    )

    results = {
        "model": "TemporalEventGNN",
        "label_field": args.label_field,
        "classes": builder.label_names,
        "test_accuracy": test_acc,
        "test_macro_f1": test_metrics["macro_f1"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_metrics["classification_report"],
        "roc_auc": roc_auc,
        "num_test_graphs": len(test_graphs),
    }

    results_path = f"{args.output_prefix}_results.json"
    with open(results_path, "w") as handle:
        json.dump(results, handle, indent=2)

    print(f"Saved evaluation results to {results_path}")
    print(f"Saved confusion matrix plot to {cm_plot_path}")
    print(f"Saved ROC curve plot to {roc_plot_path}")


if __name__ == "__main__":
    main()
