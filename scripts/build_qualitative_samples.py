#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import os
import sys

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

    predictor.load_model("Baseline", "models/baseline_model.pth")
    predictor.load_model("JellyfishNet", "models/jellyfishnet_model.pth")

    X = predictor.data_cache["X"]
    metadata = predictor.data_cache["metadata"].copy().reset_index(drop=True)
    y_true = predictor.data_cache["y"].astype(int)

    X_norm = (
        np.asarray(X, dtype=np.float32) - predictor.normalization_stats["mean"].numpy()
    ) / (predictor.normalization_stats["std"].numpy() + 1e-8)

    with torch.no_grad():
        baseline_probs = predictor.models["Baseline"](
            torch.tensor(X_norm, dtype=torch.float32)
        ).view(-1).numpy()

    with torch.no_grad():
        jelly_probs = predictor.models["JellyfishNet"](
            torch.tensor(X_norm, dtype=torch.float32)
        ).view(-1).numpy()

    baseline_pred = (baseline_probs > 0.5).astype(int)
    jelly_pred = (jelly_probs > 0.5).astype(int)

    rows = pd.DataFrame({
        "idx": np.arange(len(metadata)),
        "beach_id": metadata["beach_id"].astype(int),
        "beach_name": metadata["beach_name"].astype(str),
        "forecast_date": metadata["forecast_date"].astype(str),
        "actual": y_true,
        "baseline_prob": baseline_probs,
        "baseline_pred": baseline_pred,
        "jelly_prob": jelly_probs,
        "jelly_pred": jelly_pred,
    })
    rows["baseline_correct"] = rows["baseline_pred"] == rows["actual"]
    rows["jelly_correct"] = rows["jelly_pred"] == rows["actual"]
    rows["abs_margin_jelly"] = (rows["jelly_prob"] - 0.5).abs()

    def pick(df: pd.DataFrame, by: str | None = None, ascending: bool = False):
        if df.empty:
            return None
        if by is None:
            return df.iloc[0]
        return df.sort_values(by=by, ascending=ascending).iloc[0]

    selected = []

    r = pick(rows[(rows.actual == 1) & (rows.jelly_correct)], by="jelly_prob", ascending=False)
    if r is not None:
        selected.append(("Strong TP", r))

    r = pick(rows[(rows.actual == 0) & (rows.jelly_correct)], by="jelly_prob", ascending=True)
    if r is not None:
        selected.append(("Strong TN", r))

    r = pick(
        rows[(rows.actual == 1) & (rows.jelly_correct) & (~rows.baseline_correct)],
        by="jelly_prob",
        ascending=False,
    )
    if r is not None:
        selected.append(("Jelly wins (positive)", r))

    r = pick(
        rows[(rows.actual == 0) & (rows.jelly_correct) & (~rows.baseline_correct)],
        by="jelly_prob",
        ascending=True,
    )
    if r is not None:
        selected.append(("Jelly wins (negative)", r))

    r = pick(rows[(~rows.jelly_correct) & (~rows.baseline_correct)], by="abs_margin_jelly", ascending=True)
    if r is not None:
        selected.append(("Both wrong (ambiguous)", r))

    r = pick(rows[(~rows.jelly_correct)], by="abs_margin_jelly", ascending=False)
    if r is not None:
        selected.append(("Jelly confident miss", r))

    seen = set()
    output_rows = []
    for label, row in selected:
        sample_idx = int(row["idx"])
        if sample_idx in seen:
            continue
        seen.add(sample_idx)
        output_rows.append({
            "case_type": label,
            "beach_id": int(row["beach_id"]),
            "beach_name": row["beach_name"],
            "forecast_date": row["forecast_date"],
            "actual": int(row["actual"]),
            "baseline_prob": float(row["baseline_prob"]),
            "baseline_pred": "Yes" if int(row["baseline_pred"]) == 1 else "No",
            "baseline_correct": bool(row["baseline_correct"]),
            "jelly_prob": float(row["jelly_prob"]),
            "jelly_pred": "Yes" if int(row["jelly_pred"]) == 1 else "No",
            "jelly_correct": bool(row["jelly_correct"]),
            "prob_gap_jelly_minus_base": float(row["jelly_prob"] - row["baseline_prob"]),
        })

    out_df = pd.DataFrame(output_rows)

    summary = {
        "n_samples": int(len(rows)),
        "baseline_acc": float(rows["baseline_correct"].mean()),
        "jelly_acc": float(rows["jelly_correct"].mean()),
        "jelly_wins": int(((rows["jelly_correct"]) & (~rows["baseline_correct"])).sum()),
        "baseline_wins": int(((rows["baseline_correct"]) & (~rows["jelly_correct"])).sum()),
        "both_wrong": int(((~rows["jelly_correct"]) & (~rows["baseline_correct"])).sum()),
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    out_csv = reports_dir / "qualitative_samples.csv"
    out_df.to_csv(out_csv, index=False)

    summary_df = pd.DataFrame([summary])
    summary_csv = reports_dir / "qualitative_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"Saved: {out_csv}")
    print(out_df.to_string(index=False))
    print("\nSummary")
    print(summary_df.to_string(index=False))
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
