import json
import os
import argparse
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jellyfish.predictor import JellyfishPredictor


def _compute_metrics(actual, pred):
    tp = int(((pred == 1) & (actual == 1)).sum())
    fp = int(((pred == 1) & (actual == 0)).sum())
    fn = int(((pred == 0) & (actual == 1)).sum())
    tn = int(((pred == 0) & (actual == 0)).sum())

    acc = (tp + tn) / len(actual) if len(actual) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "n": int(len(actual)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _draw_confusion(ax, metrics, title):
    cm = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ])

    ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Neg", "Pos"], fontsize=9)
    ax.set_yticklabels(["Neg", "Pos"], fontsize=9)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Actual", fontsize=8)

    labels = [["TN", "FP"], ["FN", "TP"]]
    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for r in range(2):
        for c in range(2):
            color = "white" if cm[r, c] > threshold else "black"
            ax.text(
                c,
                r,
                f"{labels[r][c]}\n{cm[r, c]}",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=color,
            )


def main():
    parser = argparse.ArgumentParser(description="Generate Dor evaluation graphs with Baseline vs JellyfishNet")
    parser.add_argument("--start-year", type=int, default=None, help="Inclusive start year filter for report_date")
    parser.add_argument("--end-year", type=int, default=None, help="Inclusive end year filter for report_date")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix appended to output filenames")
    args = parser.parse_args()

    source_path = "reports/dor_edelist_model_test.csv"
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Missing source file: {source_path}")

    reports = pd.read_csv(source_path)
    reports["report_date"] = pd.to_datetime(reports["report_date"], errors="coerce").dt.date
    reports["model_beach_id"] = pd.to_numeric(reports["model_beach_id"], errors="coerce")

    if args.start_year is not None or args.end_year is not None:
        dt = pd.to_datetime(reports["report_date"], errors="coerce")
        mask = pd.Series(True, index=reports.index)
        if args.start_year is not None:
            mask &= dt.dt.year >= int(args.start_year)
        if args.end_year is not None:
            mask &= dt.dt.year <= int(args.end_year)
        reports = reports[mask].copy()

    if reports.empty:
        raise RuntimeError("No records found after applying year filter")

    predictor = JellyfishPredictor(device="cpu")
    predictor.load_data_cache(
        lookback_days=24,
        forecast_days=1,
        weather_csv_path=None,
        include_live_xml=False,
    )
    predictor.load_model("JellyfishNet", "models/jellyfishnet_model.pth")
    predictor.load_model("Baseline", "models/baseline_model.pth")

    rows = []
    for _, row in reports.iterrows():
        out = row.to_dict()
        beach_id = row["model_beach_id"]
        report_date = row["report_date"]

        if pd.isna(beach_id) or pd.isna(report_date):
            out["actual"] = None
            out["jellyfishnet_probability"] = None
            out["jellyfishnet_yes_no"] = None
            out["jellyfishnet_error"] = "No model beach mapping/date"
            out["baseline_probability"] = None
            out["baseline_yes_no"] = None
            out["baseline_error"] = "No model beach mapping/date"
            rows.append(out)
            continue

        out["actual"] = 1  # Dor records are all sightings

        jnet = predictor.predict_for_beach_date(int(beach_id), report_date, "JellyfishNet")
        base = predictor.predict_for_beach_date(int(beach_id), report_date, "Baseline")

        if "error" in jnet:
            out["jellyfishnet_probability"] = None
            out["jellyfishnet_yes_no"] = None
            out["jellyfishnet_error"] = jnet["error"]
        else:
            out["jellyfishnet_probability"] = float(jnet["probability"])
            out["jellyfishnet_yes_no"] = jnet["prediction"]
            out["jellyfishnet_error"] = None

        if "error" in base:
            out["baseline_probability"] = None
            out["baseline_yes_no"] = None
            out["baseline_error"] = base["error"]
        else:
            out["baseline_probability"] = float(base["probability"])
            out["baseline_yes_no"] = base["prediction"]
            out["baseline_error"] = None

        rows.append(out)

    output_df = pd.DataFrame(rows)
    valid = output_df[
        output_df["actual"].notna()
        & output_df["jellyfishnet_yes_no"].notna()
        & output_df["baseline_yes_no"].notna()
    ].copy()

    valid["actual"] = valid["actual"].astype(int)
    valid["j_pred"] = (valid["jellyfishnet_yes_no"] == "Yes").astype(int)
    valid["b_pred"] = (valid["baseline_yes_no"] == "Yes").astype(int)

    baseline_metrics = _compute_metrics(valid["actual"].values, valid["b_pred"].values)
    jelly_metrics = _compute_metrics(valid["actual"].values, valid["j_pred"].values)

    os.makedirs("reports", exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    comparison_csv = f"reports/dor_real_sightings_model_comparison_linear_baseline{suffix}.csv"
    summary_json = f"reports/dor_real_sightings_model_comparison_linear_baseline_summary{suffix}.json"
    plot_path = f"reports/dor_real_sightings_evaluation_linear_baseline{suffix}.png"

    output_df.to_csv(comparison_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "filter": {
                    "start_year": args.start_year,
                    "end_year": args.end_year,
                },
                "n_total_rows": int(len(output_df)),
                "n_valid_compared": int(len(valid)),
                "Baseline": baseline_metrics,
                "JellyfishNet": jelly_metrics,
            },
            f,
            indent=2,
        )

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Jellyfish Bloom Predictor - Evaluation on Real Sightings\n"
        "(data: meduzot.co.il / Dor profile full pull)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_metrics = fig.add_subplot(gs[0, 0])
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    baseline_vals = [
        baseline_metrics["accuracy"],
        baseline_metrics["precision"],
        baseline_metrics["recall"],
        baseline_metrics["f1"],
    ]
    jelly_vals = [
        jelly_metrics["accuracy"],
        jelly_metrics["precision"],
        jelly_metrics["recall"],
        jelly_metrics["f1"],
    ]
    x = np.arange(len(metric_names))
    width = 0.35
    bars_base = ax_metrics.bar(
        x - width / 2,
        baseline_vals,
        width,
        label="Baseline (Logistic Regression)",
        color="#4C72B0",
        alpha=0.85,
    )
    bars_jnet = ax_metrics.bar(
        x + width / 2,
        jelly_vals,
        width,
        label="JellyfishNet",
        color="#DD8452",
        alpha=0.85,
    )
    ax_metrics.set_ylim(0, 1.05)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(metric_names)
    ax_metrics.set_ylabel("Score")
    ax_metrics.set_title("Model Performance Metrics")
    ax_metrics.legend(fontsize=9)
    ax_metrics.grid(axis="y", alpha=0.3)
    for bar in list(bars_base) + list(bars_jnet):
        h = bar.get_height()
        ax_metrics.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    cm_grid = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.4)
    ax_cm_base = fig.add_subplot(cm_grid[0, 0])
    ax_cm_jnet = fig.add_subplot(cm_grid[0, 1])
    _draw_confusion(ax_cm_base, baseline_metrics, "Baseline (Logistic Regression)")
    _draw_confusion(ax_cm_jnet, jelly_metrics, "JellyfishNet")

    ax_timeline = fig.add_subplot(gs[1, 0])
    timeline = valid[["report_date", "baseline_probability", "jellyfishnet_probability", "actual"]].copy()
    timeline["report_date"] = pd.to_datetime(timeline["report_date"])
    timeline = timeline.sort_values("report_date")
    ax_timeline.plot(
        timeline["report_date"],
        timeline["baseline_probability"],
        "b-o",
        markersize=3,
        alpha=0.7,
        label="Baseline prob",
        linewidth=1,
    )
    ax_timeline.plot(
        timeline["report_date"],
        timeline["jellyfishnet_probability"],
        "r-s",
        markersize=3,
        alpha=0.7,
        label="JellyfishNet prob",
        linewidth=1,
    )
    for date_value, actual in zip(timeline["report_date"], timeline["actual"]):
        if int(actual) == 1:
            ax_timeline.axvline(date_value, color="green", alpha=0.05, linewidth=3)
    ax_timeline.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="threshold=0.5")
    ax_timeline.set_ylim(-0.05, 1.05)
    ax_timeline.set_ylabel("Predicted probability")
    ax_timeline.set_title("Prediction Timeline\n(green shading = actual sighting)")
    ax_timeline.legend(fontsize=8)
    ax_timeline.tick_params(axis="x", rotation=30)
    ax_timeline.grid(alpha=0.2)

    ax_beach = fig.add_subplot(gs[1, 1])
    beach_acc = []
    for beach_name, group in valid.groupby("model_beach_name", dropna=False):
        if pd.isna(beach_name):
            continue
        baseline_acc = float((group["b_pred"] == group["actual"]).mean())
        jelly_acc = float((group["j_pred"] == group["actual"]).mean())
        beach_acc.append((str(beach_name), baseline_acc, jelly_acc))

    beach_acc.sort(key=lambda t: t[0])
    if beach_acc:
        short_names = [name.split("-")[0][:12] for name, _, _ in beach_acc]
        baseline_acc_vals = [b for _, b, _ in beach_acc]
        jelly_acc_vals = [j for _, _, j in beach_acc]
        x_b = np.arange(len(short_names))
        w_b = 0.35

        ax_beach.bar(
            x_b - w_b / 2,
            baseline_acc_vals,
            w_b,
            label="Baseline (Logistic Regression)",
            color="#4C72B0",
            alpha=0.85,
        )
        ax_beach.bar(
            x_b + w_b / 2,
            jelly_acc_vals,
            w_b,
            label="JellyfishNet",
            color="#DD8452",
            alpha=0.85,
        )
        ax_beach.set_xticks(x_b)
        ax_beach.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
        ax_beach.set_ylim(0, 1.1)
        ax_beach.set_ylabel("Accuracy")
        ax_beach.set_title("Per-Beach Accuracy")
        ax_beach.legend(fontsize=8)
        ax_beach.grid(axis="y", alpha=0.3)

    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"valid_compared={len(valid)}")
    print(f"comparison_csv={comparison_csv}")
    print(f"summary_json={summary_json}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
