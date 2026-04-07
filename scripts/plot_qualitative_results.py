#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def main():
    reports_dir = Path("reports")
    csv_path = reports_dir / "qualitative_samples.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("qualitative_samples.csv is empty")

    labels = [
        f"{row.case_type}\nB{int(row.beach_id)} {row.beach_name}\n{row.forecast_date}"
        for _, row in df.iterrows()
    ]
    x = np.arange(len(df))

    baseline_probs = df["baseline_prob"].astype(float).values
    jelly_probs = df["jelly_prob"].astype(float).values
    actual = df["actual"].astype(int).values

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2.2, 1.3]})

    # Top panel: per-case probability comparison
    ax = axes[0]
    w = 0.36
    ax.bar(x - w / 2, baseline_probs, width=w, label="Baseline", alpha=0.85)
    ax.bar(x + w / 2, jelly_probs, width=w, label="JellyfishNet", alpha=0.85)

    # Actual label markers
    for i, y in enumerate(actual):
        marker_y = 1.03 if y == 1 else -0.03
        ax.scatter(i, marker_y, marker="*", s=120, c="black", zorder=5)

    ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.8)
    ax.set_ylim(-0.08, 1.08)
    ax.set_ylabel("Predicted probability")
    ax.set_title("Qualitative Cases: Baseline vs JellyfishNet")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)

    # Bottom panel: probability gap and correctness
    ax2 = axes[1]
    gap = df["prob_gap_jelly_minus_base"].astype(float).values

    colors = []
    for _, row in df.iterrows():
        if bool(row["jelly_correct"]) and not bool(row["baseline_correct"]):
            colors.append("#2ca02c")  # Jelly wins
        elif bool(row["baseline_correct"]) and not bool(row["jelly_correct"]):
            colors.append("#d62728")  # Baseline wins
        else:
            colors.append("#7f7f7f")  # tie (both right/both wrong)

    ax2.bar(x, gap, color=colors, alpha=0.9)
    ax2.axhline(0.0, color="black", linewidth=1)
    ax2.set_ylabel("JellyfishNet prob - Baseline prob")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.grid(axis="y", alpha=0.25)

    legend_text = (
        "Green: JellyfishNet only model correct | "
        "Red: Baseline only model correct | "
        "Gray: both correct or both wrong | "
        "Star above=actual Yes, below=actual No"
    )
    fig.text(0.5, 0.01, legend_text, ha="center", fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 1])

    out_png = reports_dir / "qualitative_results_graph.png"
    out_pdf = reports_dir / "qualitative_results_graph.pdf"
    fig.savefig(out_png, dpi=180)
    fig.savefig(out_pdf)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
