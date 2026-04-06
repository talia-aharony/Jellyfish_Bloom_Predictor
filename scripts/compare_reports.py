#!/usr/bin/env python3
"""Compare and rank training report JSON files.

Usage examples:
  python scripts/compare_reports.py
  python scripts/compare_reports.py --pattern "reports/*.json" --model Hybrid --sort-by f1
  python scripts/compare_reports.py --model LSTM --sort-by auc --top-k 10
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List


def load_report(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_rows(paths: List[str], model_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for path in paths:
        try:
            payload = load_report(path)
        except (json.JSONDecodeError, OSError):
            continue

        results = payload.get("results", {})
        config = payload.get("config", {})
        timestamp = payload.get("timestamp", "unknown")

        if model_name not in results:
            continue

        metrics = results.get(model_name, {})
        row = {
            "file": os.path.basename(path),
            "path": path,
            "timestamp": timestamp,
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "f1": float(metrics.get("f1", 0.0)),
            "auc": float(metrics.get("auc", 0.0)),
            "threshold": float(metrics.get("threshold", 0.5)),
            "val_best_recall": float(metrics.get("val_best_recall", 0.0)),
            "val_best_f1": float(metrics.get("val_best_f1", 0.0)),
            "batch_size": config.get("batch_size", "-"),
            "learning_rate": config.get("learning_rate", "-"),
            "dropout_prob": config.get("dropout_prob", "-"),
            "num_epochs": config.get("num_epochs", "-"),
            "patience": config.get("patience", "-"),
            "hybrid_hidden_dim": config.get("hybrid_hidden_dim", "-"),
        }
        rows.append(row)

    return rows


def format_row(row: Dict[str, Any]) -> str:
    return (
        f"{row['file']:<36} "
        f"{row['recall']:<8.4f} "
        f"{row['precision']:<8.4f} "
        f"{row['f1']:<8.4f} "
        f"{row['auc']:<8.4f} "
        f"{row['threshold']:<8.2f} "
        f"{row['val_best_recall']:<8.4f} "
        f"{row['val_best_f1']:<8.4f} "
        f"{row['accuracy']:<8.4f} "
        f"{str(row['learning_rate']):<10} "
        f"{str(row['dropout_prob']):<8} "
        f"{str(row['batch_size']):<6} "
        f"{str(row['hybrid_hidden_dim']):<6}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank training report JSON files")
    parser.add_argument(
        "--pattern",
        type=str,
        default="training_report*.json",
        help="Glob pattern for report files (default: training_report*.json)",
    )
    parser.add_argument(
        "--extra-pattern",
        type=str,
        default="reports/*.json",
        help="Additional glob pattern (default: reports/*.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Hybrid",
        help="Model key inside report['results'] to compare (default: Hybrid)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="recall",
        choices=["f1", "auc", "accuracy", "precision", "recall"],
        help="Metric used for ranking (default: recall)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Show top K runs (default: 20)",
    )
    args = parser.parse_args()

    paths = sorted(set(glob.glob(args.pattern) + glob.glob(args.extra_pattern)))

    if not paths:
        print("No report files found.")
        print(f"Checked patterns: {args.pattern}, {args.extra_pattern}")
        return

    rows = build_rows(paths, args.model)
    if not rows:
        print(f"No rows found for model '{args.model}' in report files.")
        return

    rows.sort(key=lambda x: x[args.sort_by], reverse=True)
    top_rows = rows[: max(args.top_k, 1)]

    print(f"\nRanking model: {args.model}")
    print(f"Sort metric: {args.sort_by}")
    print(f"Total runs considered: {len(rows)}")
    print()
    print(
        f"{'Report File':<36} {'Recall':<8} {'Prec':<8} {'F1':<8} {'AUC':<8} {'Thr':<8} {'ValRec':<8} {'ValF1':<8} {'Acc':<8} {'LR':<10} {'Dropout':<8} {'Batch':<6} {'HDim':<6}"
    )
    print("-" * 148)
    for row in top_rows:
        print(format_row(row))

    best = top_rows[0]
    print("\nBest run summary")
    print("-" * 40)
    print(f"File: {best['file']}")
    print(f"Path: {best['path']}")
    print(f"Timestamp: {best['timestamp']}")
    print(
        f"Recall: {best['recall']:.4f}, Precision: {best['precision']:.4f}, "
        f"F1: {best['f1']:.4f}"
    )
    print(f"Accuracy: {best['accuracy']:.4f}, AUC: {best['auc']:.4f}")
    print(
        f"Threshold: {best['threshold']:.2f}, "
        f"Validation-best Recall: {best['val_best_recall']:.4f}, "
        f"Validation-best F1: {best['val_best_f1']:.4f}"
    )
    print(
        "Config: "
        f"lr={best['learning_rate']}, "
        f"dropout={best['dropout_prob']}, "
        f"batch={best['batch_size']}, "
        f"hidden_dim={best['hybrid_hidden_dim']}, "
        f"epochs={best['num_epochs']}, "
        f"patience={best['patience']}"
    )


if __name__ == "__main__":
    main()
