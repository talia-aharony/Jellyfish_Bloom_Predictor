#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    reports_dir = Path("reports")
    csv_path = reports_dir / "jellynet_per_beach_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1) Per-beach metric bars (F1/Recall/Precision)
    metric_df = df.sort_values("f1", ascending=False).copy()
    labels = [f"{int(bid)} {name}" for bid, name in zip(metric_df["beach_id"], metric_df["beach_name"])]
    x = np.arange(len(metric_df))
    w = 0.26

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - w, metric_df["precision"], width=w, label="Precision")
    ax.bar(x, metric_df["recall"], width=w, label="Recall")
    ax.bar(x + w, metric_df["f1"], width=w, label="F1")
    ax.set_ylim(0, 1)
    ax.set_title("JellyfishNet per-beach metrics (sorted by F1)")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out1_png = reports_dir / "jellynet_per_beach_metrics.png"
    out1_pdf = reports_dir / "jellynet_per_beach_metrics.pdf"
    fig.savefig(out1_png, dpi=180)
    fig.savefig(out1_pdf)
    plt.close(fig)

    # 2) Calibration-style plot: actual vs predicted positive rate per beach
    fig, ax = plt.subplots(figsize=(8, 7))
    sizes = 20 + 3 * df["n_samples"].to_numpy()
    ax.scatter(df["actual_positive_rate"], df["pred_positive_rate"], s=sizes, alpha=0.75)

    for _, row in df.iterrows():
        label = f"{int(row['beach_id'])} {row['beach_name']}"
        ax.annotate(
            label,
            (row["actual_positive_rate"], row["pred_positive_rate"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )

    lo = min(df["actual_positive_rate"].min(), df["pred_positive_rate"].min())
    hi = max(df["actual_positive_rate"].max(), df["pred_positive_rate"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    ax.set_title("JellyfishNet per-beach predicted vs actual positive rate")
    ax.set_xlabel("Actual positive rate")
    ax.set_ylabel("Predicted positive rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out2_png = reports_dir / "jellynet_per_beach_calibration.png"
    out2_pdf = reports_dir / "jellynet_per_beach_calibration.pdf"
    fig.savefig(out2_png, dpi=180)
    fig.savefig(out2_pdf)
    plt.close(fig)

    # 3) Confusion composition by beach (normalized)
    comp = df.copy()
    comp["tp_rate"] = comp["tp"] / comp["n_samples"]
    comp["fp_rate"] = comp["fp"] / comp["n_samples"]
    comp["tn_rate"] = comp["tn"] / comp["n_samples"]
    comp["fn_rate"] = comp["fn"] / comp["n_samples"]
    comp = comp.sort_values("f1", ascending=False)

    labels = [
        f"{int(bid)}\n{name}"
        for bid, name in zip(comp["beach_id"], comp["beach_name"])
    ]
    x = np.arange(len(comp))

    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x, comp["tn_rate"], label="TN rate")
    b2 = ax.bar(x, comp["fp_rate"], bottom=comp["tn_rate"], label="FP rate")
    b3 = ax.bar(x, comp["fn_rate"], bottom=comp["tn_rate"] + comp["fp_rate"], label="FN rate")
    b4 = ax.bar(
        x,
        comp["tp_rate"],
        bottom=comp["tn_rate"] + comp["fp_rate"] + comp["fn_rate"],
        label="TP rate",
    )

    ax.set_ylim(0, 1)
    ax.set_title("JellyfishNet confusion composition per beach (normalized)")
    ax.set_ylabel("Fraction of samples")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_xlabel("Beach ID + name (sorted by F1)")
    ax.legend(ncol=4, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out3_png = reports_dir / "jellynet_per_beach_confusion.png"
    out3_pdf = reports_dir / "jellynet_per_beach_confusion.pdf"
    fig.savefig(out3_png, dpi=180)
    fig.savefig(out3_pdf)
    plt.close(fig)

    print(f"Saved: {out1_png}")
    print(f"Saved: {out1_pdf}")
    print(f"Saved: {out2_png}")
    print(f"Saved: {out2_pdf}")
    print(f"Saved: {out3_png}")
    print(f"Saved: {out3_pdf}")


if __name__ == "__main__":
    main()
