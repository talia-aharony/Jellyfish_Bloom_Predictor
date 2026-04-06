#!/usr/bin/env python3
"""Check JellyfishNet predictions on the training data it was fit on.

This is an in-sample sanity check, not a true new-data evaluation.
It produces:
- overall metrics on the cached dataset
- a small set of representative examples with beach names
- a simple graph for report use
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jellyfish.predictor import JellyfishPredictor


def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    predictor = JellyfishPredictor(device="cpu")
    predictor.load_data_cache(
        lookback_days=24,
        forecast_days=1,
        weather_csv_path=None,
        include_live_xml=True,
    )
    predictor.load_model("JellyfishNet", "models/jellyfishnet_model.pth")

    X = predictor.data_cache["X"].astype(np.float32)
    y_true = predictor.data_cache["y"].astype(int)
    metadata = predictor.data_cache["metadata"].copy().reset_index(drop=True)

    X_norm = (X - predictor.normalization_stats["mean"].numpy()) / (
        predictor.normalization_stats["std"].numpy() + 1e-8
    )

    with torch.no_grad():
        probs = predictor.models["JellyfishNet"](
            torch.tensor(X_norm, dtype=torch.float32)
        ).view(-1).cpu().numpy()

    preds = (probs > 0.5).astype(int)

    overall = {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    results = pd.DataFrame(
        {
            "beach_id": metadata["beach_id"].astype(int),
            "beach_name": metadata["beach_name"].astype(str),
            "forecast_date": metadata["forecast_date"].astype(str),
            "actual": y_true,
            "probability": probs,
            "prediction": np.where(preds == 1, "Yes", "No"),
            "correct": preds == y_true,
            "confidence_distance": np.abs(probs - 0.5),
        }
    )

    # Representative examples for a report
    positives = results[results["actual"] == 1].copy().sort_values(["correct", "probability"], ascending=[False, False])
    negatives = results[results["actual"] == 0].copy().sort_values(["correct", "probability"], ascending=[False, True])

    true_pos = positives[positives["correct"]].head(4)
    false_neg = positives[~positives["correct"]].sort_values("confidence_distance", ascending=False).head(3)
    true_neg = negatives[negatives["correct"]].head(4)
    false_pos = negatives[~negatives["correct"]].sort_values("confidence_distance", ascending=False).head(3)

    sample_table = pd.concat(
        [
            true_pos.assign(case_type="True positive"),
            false_neg.assign(case_type="False negative"),
            true_neg.assign(case_type="True negative"),
            false_pos.assign(case_type="False positive"),
        ],
        ignore_index=True,
    )[
        [
            "case_type",
            "beach_id",
            "beach_name",
            "forecast_date",
            "actual",
            "probability",
            "prediction",
            "correct",
            "confidence_distance",
        ]
    ]

    # Save tables
    out_csv = reports_dir / "jellynet_training_data_check_samples.csv"
    sample_table.to_csv(out_csv, index=False)

    summary_csv = reports_dir / "jellynet_training_data_check_summary.csv"
    pd.DataFrame([
        {
            **overall,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    ]).to_csv(summary_csv, index=False)

    # Graph 1: probability distribution by actual class
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.hist(probs[y_true == 0], bins=24, alpha=0.7, label="Actual No", color="#1f77b4")
    ax.hist(probs[y_true == 1], bins=24, alpha=0.7, label="Actual Yes", color="#ff7f0e")
    ax.axvline(0.5, linestyle="--", color="black", linewidth=1)
    ax.set_title("JellyfishNet on training data: predicted probabilities")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    # Graph 2: confusion matrix
    ax = axes[1]
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix on training data")
    ax.set_xticks([0, 1], ["Pred No", "Pred Yes"])
    ax.set_yticks([0, 1], ["Actual No", "Actual Yes"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out_png = reports_dir / "jellynet_training_data_check.png"
    out_pdf = reports_dir / "jellynet_training_data_check.pdf"
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)
    plt.close(fig)

    print("\nTRAINING-DATA CHECK (IN-SAMPLE ONLY)")
    print(pd.DataFrame([overall]).to_string(index=False))
    print(f"tp={tp}, fp={fp}, tn={tn}, fn={fn}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print("\nRepresentative examples:")
    print(sample_table.to_string(index=False))


if __name__ == "__main__":
    main()
