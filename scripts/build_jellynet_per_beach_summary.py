#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jellyfish.predictor import JellyfishPredictor


def main():
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
    feature_cols = list(predictor.data_cache["feature_cols"])

    X_norm = (X - predictor.normalization_stats["mean"].numpy()) / (
        predictor.normalization_stats["std"].numpy() + 1e-8
    )

    with torch.no_grad():
        probs = predictor.models["JellyfishNet"](
            torch.tensor(X_norm, dtype=torch.float32)
        ).view(-1).cpu().numpy()

    preds = (probs > 0.5).astype(int)
    conf_dist = np.abs(probs - 0.5)

    df = pd.DataFrame(
        {
            "beach_id": metadata["beach_id"].astype(int),
            "beach_name": metadata["beach_name"].astype(str),
            "actual": y_true,
            "pred": preds,
            "probability": probs,
            "confidence_distance": conf_dist,
        }
    )

    # last-day contextual inputs from lookback window
    idx_month = feature_cols.index("month") if "month" in feature_cols else None
    idx_diameter = feature_cols.index("diameter_cm") if "diameter_cm" in feature_cols else None
    idx_obs_count = feature_cols.index("observation_count") if "observation_count" in feature_cols else None
    idx_sting = feature_cols.index("sting") if "sting" in feature_cols else None

    last_day = X[:, -1, :]
    if idx_month is not None:
        df["month_last"] = last_day[:, idx_month]
    else:
        df["month_last"] = np.nan

    if idx_diameter is not None:
        df["diameter_cm_last"] = last_day[:, idx_diameter]
    else:
        df["diameter_cm_last"] = np.nan

    if idx_obs_count is not None:
        df["observation_count_last"] = last_day[:, idx_obs_count]
    else:
        df["observation_count_last"] = np.nan

    if idx_sting is not None:
        df["sting_last"] = last_day[:, idx_sting]
    else:
        df["sting_last"] = np.nan

    rows = []
    for (beach_id, beach_name), g in df.groupby(["beach_id", "beach_name"], sort=True):
        y = g["actual"].to_numpy(dtype=int)
        yhat = g["pred"].to_numpy(dtype=int)

        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())

        rows.append(
            {
                "beach_id": int(beach_id),
                "beach_name": str(beach_name),
                "n_samples": int(len(g)),
                "actual_positive_rate": float(g["actual"].mean()),
                "pred_positive_rate": float(g["pred"].mean()),
                "avg_probability": float(g["probability"].mean()),
                "avg_confidence_distance": float(g["confidence_distance"].mean()),
                "avg_month_last": float(g["month_last"].mean()),
                "avg_diameter_cm_last": float(g["diameter_cm_last"].mean()),
                "avg_observation_count_last": float(g["observation_count_last"].mean()),
                "avg_sting_last": float(g["sting_last"].mean()),
                "accuracy": float(accuracy_score(y, yhat)),
                "precision": float(precision_score(y, yhat, zero_division=0)),
                "recall": float(recall_score(y, yhat, zero_division=0)),
                "f1": float(f1_score(y, yhat, zero_division=0)),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )

    out = pd.DataFrame(rows).sort_values("beach_id").reset_index(drop=True)

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    out_csv = reports_dir / "jellynet_per_beach_summary.csv"
    out.to_csv(out_csv, index=False)

    # Add a header text file with metadata context for report
    header_txt = reports_dir / "jellynet_per_beach_summary_header.txt"
    header_txt.write_text(
        "# Per-beach aggregated qualitative/quantitative summary\n"
        "# Primary model: JellyfishNet\n"
        "# Threshold used for predictions: 0.5\n"
        "# Each row aggregates all test samples for a beach.\n"
        "# Metrics: accuracy/precision/recall/f1, confusion counts (tp/fp/tn/fn),\n"
        "# and average contextual inputs from the lookback last day.\n",
        encoding="utf-8",
    )

    print(f"Saved: {out_csv}")
    print(out.to_string(index=False))
    print(f"Saved: {header_txt}")


if __name__ == "__main__":
    main()
