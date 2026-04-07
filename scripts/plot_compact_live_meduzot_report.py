import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_metrics(actual, pred):
    tp = int(((pred == 1) & (actual == 1)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())
    tn = int(((pred == 0) & (actual == 0)).sum())

    n = len(actual)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "n": int(n),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main():
    src = "reports/all_real_sightings_model_comparison_linear_baseline_2024_06_2026_live_meduzot.csv"
    out_png = "reports/all_real_sightings_compact_report_2024_06_2026_live_meduzot.png"
    out_pdf = "reports/all_real_sightings_compact_report_2024_06_2026_live_meduzot.pdf"
    out_json = "reports/all_real_sightings_compact_report_2024_06_2026_live_meduzot_summary.json"

    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing input CSV: {src}")

    df = pd.read_csv(src)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    valid = df[
        df["actual"].notna()
        & df["baseline_yes_no"].notna()
        & df["jellyfishnet_yes_no"].notna()
        & df["report_date"].notna()
    ].copy()

    valid["actual"] = valid["actual"].astype(int)
    valid["b_pred"] = (valid["baseline_yes_no"] == "Yes").astype(int)
    valid["j_pred"] = (valid["jellyfishnet_yes_no"] == "Yes").astype(int)

    m_b = compute_metrics(valid["actual"].values, valid["b_pred"].values)
    m_j = compute_metrics(valid["actual"].values, valid["j_pred"].values)

    fig = plt.figure(figsize=(12.5, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)

    fig.suptitle(
        "Jellyfish Bloom Predictor - Compact Real-Sightings Report\n"
        "Live meduzot pull, 2024-06 to 2026-04",
        fontsize=14,
        fontweight="bold",
    )

    ax0 = fig.add_subplot(gs[0, 0])
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    b_vals = [m_b["accuracy"], m_b["precision"], m_b["recall"], m_b["f1"]]
    j_vals = [m_j["accuracy"], m_j["precision"], m_j["recall"], m_j["f1"]]
    x = np.arange(len(metric_names))
    w = 0.34

    bars_b = ax0.bar(x - w / 2, b_vals, w, label="Baseline (Logistic Regression)", color="#4C72B0")
    bars_j = ax0.bar(x + w / 2, j_vals, w, label="JellyfishNet", color="#DD8452")
    ax0.set_ylim(0, 1.08)
    ax0.set_xticks(x)
    ax0.set_xticklabels(metric_names)
    ax0.set_title("Performance Metrics")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(fontsize=8, loc="lower left")

    for bars in [bars_b, bars_j]:
        for b in bars:
            h = b.get_height()
            ax0.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax1 = fig.add_subplot(gs[0, 1])
    models = ["Baseline", "JellyfishNet"]
    fn_counts = [m_b["fn"], m_j["fn"]]
    tp_counts = [m_b["tp"], m_j["tp"]]
    x2 = np.arange(len(models))

    ax1.bar(x2, fn_counts, label="FN (missed sightings)", color="#d62728", alpha=0.85)
    ax1.bar(x2, tp_counts, bottom=fn_counts, label="TP (detected sightings)", color="#2ca02c", alpha=0.85)
    ax1.set_xticks(x2)
    ax1.set_xticklabels(models)
    ax1.set_title("Outcome Breakdown (All labels are Positive)")
    ax1.set_ylabel("Count")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(axis="y", alpha=0.25)

    for i, (fn, tp) in enumerate(zip(fn_counts, tp_counts)):
        total = fn + tp
        miss_pct = (fn / total * 100) if total else 0
        hit_pct = (tp / total * 100) if total else 0
        ax1.text(i, fn / 2 if fn else 0.5, f"FN\n{fn}\n{miss_pct:.1f}%", ha="center", va="center", fontsize=8, color="white" if fn > 20 else "black")
        ax1.text(i, fn + tp / 2 if tp else fn + 0.5, f"TP\n{tp}\n{hit_pct:.1f}%", ha="center", va="center", fontsize=8, color="white" if tp > 20 else "black")

    ax1.text(
        0.02,
        -0.24,
        "TN/FP are structurally 0 here because this set contains only reported sightings (no negative ground-truth events).",
        transform=ax1.transAxes,
        fontsize=8,
        color="#444444",
    )

    fig.savefig(out_png, dpi=170)
    fig.savefig(out_pdf)
    plt.close(fig)

    summary = {
        "n_total_rows": int(len(df)),
        "n_valid_compared": int(len(valid)),
        "date_min": str(valid["report_date"].min().date()) if len(valid) else None,
        "date_max": str(valid["report_date"].max().date()) if len(valid) else None,
        "Baseline": m_b,
        "JellyfishNet": m_j,
        "note": "All labels are positive sightings; TN/FP are expected to be zero.",
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"png={out_png}")
    print(f"pdf={out_pdf}")
    print(f"summary={out_json}")
    print(f"n_valid_compared={len(valid)}")


if __name__ == "__main__":
    main()
