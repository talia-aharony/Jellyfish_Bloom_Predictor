#!/usr/bin/env python3
"""Generate qualitative model analysis report with concrete sample outputs.

The script creates:
1) Markdown report with narrative + sample cases
2) CSV with selected qualitative examples

Usage:
  python scripts/qualitative_analysis.py
    python scripts/qualitative_analysis.py --models GRU,Hybrid --samples-per-category 20
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jellyfish.predictor import JellyfishPredictor, create_engineered_features_forecasting
from jellyfish.settings import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_USE_INTEGRATED_DATA,
    DEFAULT_WEATHER_CSV_PATH,
    DEFAULT_INCLUDE_LIVE_XML,
)


def classification_metrics(y_true, probs, threshold=0.5):
    y_pred = (probs >= threshold).astype(int)

    acc = float(np.mean(y_pred == y_true))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "y_pred": y_pred,
    }


def classification_metrics_from_preds(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(np.mean(y_pred == y_true)) if len(y_true) > 0 else 0.0
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def infer_probabilities_for_indices(predictor, model_name, indices):
    probs = []
    X_all = predictor.data_cache["X"]

    for idx in indices:
        seq = X_all[idx]
        if model_name == "Baseline":
            X_eng = create_engineered_features_forecasting(
                seq[np.newaxis, ...],
                lookback=predictor.normalization_stats["lookback_days"],
            )
            X_eng_tensor = torch.FloatTensor(X_eng[0])
            mean_eng = predictor.normalization_stats["mean_eng"]
            std_eng = predictor.normalization_stats["std_eng"]
            X_input = (X_eng_tensor - mean_eng) / (std_eng + 1e-8)
        else:
            X_tensor = torch.FloatTensor(seq)
            mean = predictor.normalization_stats["mean"]
            std = predictor.normalization_stats["std"]
            X_input = (X_tensor - mean) / (std + 1e-8)

        prob = predictor.predict_sequence(X_input, model_name)
        probs.append(prob)

    return np.array(probs, dtype=np.float32)


def subset_indices_like_training_split(n_total):
    train_size = int(0.7 * n_total)
    val_size = int(0.15 * n_total)
    test_size = n_total - train_size - val_size

    full = torch.arange(n_total)
    train_set, val_set, test_set = torch.utils.data.random_split(
        full,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    return np.array(test_set.indices, dtype=int)


def render_markdown_table(df, columns):
    if df.empty:
        return "_No samples found._\n"

    view = df[columns].copy()
    for col in ["probability", "confidence_distance", "gru_prob", "hybrid_prob"]:
        if col in view.columns:
            view[col] = view[col].astype(float).map(lambda x: f"{x:.3f}")

    try:
        return view.to_markdown(index=False) + "\n"
    except ImportError:
        return "```\n" + view.to_string(index=False) + "\n```\n"


def dataframe_to_markdown_or_text(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def csv_comment_header(primary_model, model_names, threshold, samples_per_category):
    lines = [
        "# Qualitative sample outputs",
        f"# Primary model for case labels: {primary_model}",
        f"# Models included: {', '.join(model_names)}",
        f"# Classification threshold: {threshold}",
        f"# Requested samples per category: {samples_per_category} (max total rows ≈ 4 x this value)",
        "# case_type legend: TP=true positive, TN=true negative, FP=false positive, FN=false negative",
        "# column guide:",
        "# - sample_index: row index in full sequence dataset",
        "# - beach_id, beach_name, forecast_date: sample identity",
        "# - actual: ground truth label (1 bloom, 0 no bloom)",
        "# - probability: primary-model bloom probability",
        "# - prediction: thresholded class from primary-model probability",
        "# - confidence_distance: abs(probability - threshold)",
        "# - month_last, diameter_cm_last, observation_count_last, sting_last: last-day context from lookback window",
        "# - <model>_prob columns: comparison probabilities from each model",
    ]
    return "\n".join(lines) + "\n"


def beach_csv_comment_header(primary_model, threshold):
    lines = [
        "# Per-beach aggregated qualitative/quantitative summary",
        f"# Primary model: {primary_model}",
        f"# Threshold used for predictions: {threshold}",
        "# Each row aggregates all test samples for a beach.",
        "# Metrics: accuracy/precision/recall/f1, confusion counts (tp/fp/tn/fn),",
        "# and average contextual inputs from the lookback last day.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative analysis report with sample model outputs")
    parser.add_argument("--models", type=str, default="GRU,Hybrid", help="Comma-separated model names to analyze")
    parser.add_argument("--samples-per-category", type=int, default=20)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--use-integrated-data", action="store_true", default=DEFAULT_USE_INTEGRATED_DATA)
    parser.add_argument("--weather-csv-path", type=str, default=DEFAULT_WEATHER_CSV_PATH)
    parser.add_argument("--disable-live-xml", action="store_true", default=not DEFAULT_INCLUDE_LIVE_XML)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-md", type=str, default="reports/qualitative_analysis.md")
    parser.add_argument("--output-csv", type=str, default="reports/qualitative_samples.csv")
    parser.add_argument("--output-beach-csv", type=str, default="reports/qualitative_by_beach.csv")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        raise ValueError("At least one model must be provided in --models")

    os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_beach_csv) or ".", exist_ok=True)

    predictor = JellyfishPredictor(device="cpu")
    predictor.load_data_cache(
        lookback_days=args.lookback_days,
        forecast_days=1,
        use_integrated_data=args.use_integrated_data,
        weather_csv_path=args.weather_csv_path,
        include_live_xml=not args.disable_live_xml,
    )

    for model_name in model_names:
        model_path = f"{model_name.lower()}_model.pth"
        predictor.load_model(model_name, model_path)

    metadata = predictor.data_cache["metadata"].copy().reset_index(drop=True)
    y_all = predictor.data_cache["y"].astype(int)
    X_all = predictor.data_cache["X"]

    test_indices = subset_indices_like_training_split(len(metadata))
    test_meta = metadata.iloc[test_indices].copy().reset_index(drop=True)
    y_true = y_all[test_indices]

    probs_by_model = {}
    metrics_by_model = {}

    for model_name in model_names:
        probs = infer_probabilities_for_indices(predictor, model_name, test_indices)
        probs_by_model[model_name] = probs
        metrics_by_model[model_name] = classification_metrics(y_true, probs, threshold=args.threshold)

    primary_model = model_names[0]
    primary_probs = probs_by_model[primary_model]
    primary_preds = metrics_by_model[primary_model]["y_pred"]

    feature_idx = predictor._feature_indices()

    qual_df = pd.DataFrame(
        {
            "sample_index": test_indices,
            "beach_id": test_meta["beach_id"].astype(int).values,
            "beach_name": test_meta["beach_name"].astype(str).values,
            "forecast_date": test_meta["forecast_date"].astype(str).values,
            "actual": y_true,
            "probability": primary_probs,
            "prediction": primary_preds,
            "confidence_distance": np.abs(primary_probs - args.threshold),
            "month_last": X_all[test_indices, -1, feature_idx["month"]],
            "diameter_cm_last": X_all[test_indices, -1, feature_idx["diameter_cm"]],
            "observation_count_last": X_all[test_indices, -1, feature_idx["observation_count"]],
            "sting_last": X_all[test_indices, -1, feature_idx["sting"]],
        }
    )

    qual_df["case_type"] = ""
    qual_df.loc[(qual_df["actual"] == 1) & (qual_df["prediction"] == 1), "case_type"] = "TP"
    qual_df.loc[(qual_df["actual"] == 0) & (qual_df["prediction"] == 0), "case_type"] = "TN"
    qual_df.loc[(qual_df["actual"] == 0) & (qual_df["prediction"] == 1), "case_type"] = "FP"
    qual_df.loc[(qual_df["actual"] == 1) & (qual_df["prediction"] == 0), "case_type"] = "FN"

    selected_rows = []
    for ctype in ["TP", "TN", "FP", "FN"]:
        part = qual_df[qual_df["case_type"] == ctype].sort_values("confidence_distance", ascending=False)
        selected_rows.append(part.head(args.samples_per_category))

    selected_df = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()

    beach_rows = []
    for (beach_id, beach_name), group in qual_df.groupby(["beach_id", "beach_name"], sort=True):
        metrics = classification_metrics_from_preds(group["actual"].to_numpy(), group["prediction"].to_numpy())
        beach_rows.append(
            {
                "beach_id": int(beach_id),
                "beach_name": str(beach_name),
                "n_samples": int(len(group)),
                "actual_positive_rate": float(group["actual"].mean()),
                "pred_positive_rate": float(group["prediction"].mean()),
                "avg_probability": float(group["probability"].mean()),
                "avg_confidence_distance": float(group["confidence_distance"].mean()),
                "avg_month_last": float(group["month_last"].mean()),
                "avg_diameter_cm_last": float(group["diameter_cm_last"].mean()),
                "avg_observation_count_last": float(group["observation_count_last"].mean()),
                "avg_sting_last": float(group["sting_last"].mean()),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "tn": metrics["tn"],
                "fn": metrics["fn"],
            }
        )

    beach_df = pd.DataFrame(beach_rows)
    if not beach_df.empty:
        beach_df = beach_df.sort_values(["f1", "accuracy", "n_samples"], ascending=[False, False, False]).reset_index(drop=True)

    disagreement_df = pd.DataFrame()
    if len(model_names) >= 2:
        m1, m2 = model_names[0], model_names[1]
        p1 = probs_by_model[m1]
        p2 = probs_by_model[m2]
        y1 = (p1 >= args.threshold).astype(int)
        y2 = (p2 >= args.threshold).astype(int)
        diff_mask = y1 != y2

        disagreement_df = test_meta.loc[diff_mask, ["beach_id", "beach_name", "forecast_date"]].copy()
        disagreement_df["actual"] = y_true[diff_mask]
        disagreement_df[f"{m1.lower()}_prob"] = p1[diff_mask]
        disagreement_df[f"{m2.lower()}_prob"] = p2[diff_mask]
        disagreement_df["prob_gap"] = np.abs(disagreement_df[f"{m1.lower()}_prob"] - disagreement_df[f"{m2.lower()}_prob"])
        disagreement_df = disagreement_df.sort_values("prob_gap", ascending=False).head(args.samples_per_category)

        idx_to_pos = {int(idx): pos for pos, idx in enumerate(test_indices)}
        selected_positions = selected_df["sample_index"].astype(int).map(idx_to_pos)
        selected_df[f"{m1.lower()}_prob"] = p1[selected_positions.to_numpy(dtype=int)]
        selected_df[f"{m2.lower()}_prob"] = p2[selected_positions.to_numpy(dtype=int)]

    with open(args.output_csv, "w", encoding="utf-8") as f:
        f.write(csv_comment_header(primary_model, model_names, args.threshold, args.samples_per_category))
    selected_df.to_csv(args.output_csv, index=False, mode="a")

    with open(args.output_beach_csv, "w", encoding="utf-8") as f:
        f.write(beach_csv_comment_header(primary_model, args.threshold))
    beach_df.to_csv(args.output_beach_csv, index=False, mode="a")

    lines = []
    lines.append("# Qualitative Model Analysis")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Lookback days: {args.lookback_days}")
    lines.append(f"Integrated data: {bool(args.use_integrated_data)}")
    lines.append(f"Primary model for case breakdown: {primary_model}")
    lines.append("")

    lines.append("## Quantitative Context")
    lines.append("")
    metrics_table = []
    for m in model_names:
        mm = metrics_by_model[m]
        metrics_table.append(
            {
                "model": m,
                "accuracy": mm["accuracy"],
                "precision": mm["precision"],
                "recall": mm["recall"],
                "f1": mm["f1"],
                "tp": mm["tp"],
                "fp": mm["fp"],
                "tn": mm["tn"],
                "fn": mm["fn"],
            }
        )
    lines.append(dataframe_to_markdown_or_text(pd.DataFrame(metrics_table)))
    lines.append("")

    lines.append("## Sample Outputs (Most Interpretable Cases)")
    lines.append("")
    lines.append(
        "The following samples show concrete model outputs and put performance in context. "
        "Cases are selected by highest confidence distance from threshold."
    )
    lines.append("")

    for ctype in ["TP", "TN", "FP", "FN"]:
        lines.append(f"### {ctype} examples")
        lines.append("")
        part = selected_df[selected_df["case_type"] == ctype]
        cols = [
            "beach_id",
            "beach_name",
            "forecast_date",
            "actual",
            "prediction",
            "probability",
            "confidence_distance",
            "month_last",
            "observation_count_last",
            "diameter_cm_last",
            "sting_last",
        ]
        lines.append(render_markdown_table(part, [c for c in cols if c in part.columns]))

    if not disagreement_df.empty:
        lines.append("## Model Disagreement Cases")
        lines.append("")
        lines.append(
            f"These are examples where {model_names[0]} and {model_names[1]} produce different class predictions, "
            "which helps explain strengths/weaknesses by input type."
        )
        lines.append("")
        lines.append(dataframe_to_markdown_or_text(disagreement_df))
        lines.append("")

    lines.append("## Performance By Beach (Aggregated)")
    lines.append("")
    lines.append(
        "The table below aggregates performance and context per beach across the full test subset, "
        "so you can see where the model predicts well vs poorly by location."
    )
    lines.append("")
    beach_cols = [
        "beach_id",
        "beach_name",
        "n_samples",
        "actual_positive_rate",
        "pred_positive_rate",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "tn",
        "fn",
        "avg_probability",
        "avg_confidence_distance",
        "avg_observation_count_last",
        "avg_diameter_cm_last",
        "avg_sting_last",
    ]
    if beach_df.empty:
        lines.append("_No beach-level samples found._")
    else:
        lines.append(dataframe_to_markdown_or_text(beach_df[[c for c in beach_cols if c in beach_df.columns]]))
        lines.append("")
        lines.append(f"Beach-level CSV saved to: {args.output_beach_csv}")
        lines.append("")

    mm = metrics_by_model[primary_model]
    if mm["fn"] > mm["fp"]:
        error_note = "The primary model misses more positives than negatives (higher FN), suggesting conservative bloom detection."
    elif mm["fp"] > mm["fn"]:
        error_note = "The primary model raises more false alarms than misses (higher FP), suggesting aggressive bloom detection."
    else:
        error_note = "The primary model has balanced FP/FN counts in this split."

    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append(f"- {error_note}")
    lines.append("- TP/TN samples illustrate where the model is confident and likely exploiting stable seasonal/observation patterns.")
    lines.append("- FP/FN samples highlight specific input conditions where the model underperforms and may require feature or threshold tuning.")
    lines.append("")

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✓ Wrote qualitative markdown report: {args.output_md}")
    print(f"✓ Wrote qualitative sample CSV: {args.output_csv}")
    print(f"✓ Wrote beach-level qualitative CSV: {args.output_beach_csv}")


if __name__ == "__main__":
    main()
