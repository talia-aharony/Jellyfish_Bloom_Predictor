import pandas as pd

from jellyfish.predictor import JellyfishPredictor
from scripts.evaluate_dor_reports import get_beach_mapping, parse_dor_reports


def _extract_prediction_fields(result):
    if "error" in result:
        return {
            "probability": None,
            "percentage": None,
            "yes_no": None,
            "confidence": None,
            "error": result["error"],
            "extrapolated": result.get("extrapolated"),
            "extrapolated_from_date": result.get("extrapolated_from_date"),
        }

    return {
        "probability": float(result["probability"]),
        "percentage": float(result["percentage"]),
        "yes_no": result["prediction"],
        "confidence": result["confidence"],
        "error": None,
        "extrapolated": result.get("extrapolated"),
        "extrapolated_from_date": result.get("extrapolated_from_date"),
    }


def _summarize_model(df, prefix):
    valid = df[f"{prefix}_yes_no"].notna()
    valid_count = int(valid.sum())
    yes_count = int((df[f"{prefix}_yes_no"] == "Yes").sum())
    no_count = int((df[f"{prefix}_yes_no"] == "No").sum())
    hit_rate = (yes_count / valid_count) * 100.0 if valid_count else 0.0

    return {
        "model": prefix,
        "predictions": valid_count,
        "pred_yes": yes_count,
        "pred_no": no_count,
        "hit_rate_on_real_sightings_pct": hit_rate,
    }


def main():
    reports_df = parse_dor_reports()
    print(f"Parsed Dor reports: {len(reports_df)}")

    reports_df["model_beach_id"] = reports_df["meduzot_beach_id"].map(get_beach_mapping())

    predictor = JellyfishPredictor(device="cpu")
    predictor.load_data_cache(
        lookback_days=24,
        forecast_days=1,
        weather_csv_path=None,
        include_live_xml=False,
    )
    predictor.load_model("JellyfishNet", "models/jellyfishnet_model.pth")
    predictor.load_model("Baseline", "models/baseline_model.pth")

    metadata = predictor.data_cache["metadata"][["beach_id", "beach_name"]].drop_duplicates()
    metadata = metadata.rename(
        columns={"beach_id": "model_beach_id", "beach_name": "model_beach_name"}
    )
    reports_df = reports_df.merge(metadata, on="model_beach_id", how="left")

    rows = []
    for _, row in reports_df.iterrows():
        out = row.to_dict()
        model_beach_id = row["model_beach_id"]

        if pd.isna(model_beach_id):
            for prefix in ["jellyfishnet", "baseline"]:
                out[f"{prefix}_probability"] = None
                out[f"{prefix}_percentage"] = None
                out[f"{prefix}_yes_no"] = None
                out[f"{prefix}_confidence"] = None
                out[f"{prefix}_error"] = "No model beach mapping"
                out[f"{prefix}_extrapolated"] = None
                out[f"{prefix}_extrapolated_from_date"] = None
            rows.append(out)
            continue

        jelly_result = predictor.predict_for_beach_date(
            int(model_beach_id), row["report_date"], "JellyfishNet"
        )
        base_result = predictor.predict_for_beach_date(
            int(model_beach_id), row["report_date"], "Baseline"
        )

        jelly = _extract_prediction_fields(jelly_result)
        base = _extract_prediction_fields(base_result)

        for k, v in jelly.items():
            out[f"jellyfishnet_{k}"] = v
        for k, v in base.items():
            out[f"baseline_{k}"] = v

        if jelly["yes_no"] is not None and base["yes_no"] is not None:
            out["both_agree"] = jelly["yes_no"] == base["yes_no"]
            out["jellyfishnet_minus_baseline_pct"] = jelly["percentage"] - base["percentage"]
        else:
            out["both_agree"] = None
            out["jellyfishnet_minus_baseline_pct"] = None

        rows.append(out)

    output_df = pd.DataFrame(rows)
    output_df = output_df.sort_values(["report_date", "report_time", "observation_id"])

    comparison_summary = pd.DataFrame(
        [
            _summarize_model(output_df, "jellyfishnet"),
            _summarize_model(output_df, "baseline"),
        ]
    )

    agreement_df = output_df[
        output_df["jellyfishnet_yes_no"].notna() & output_df["baseline_yes_no"].notna()
    ]
    agreement_rate = (
        float((agreement_df["both_agree"] == True).mean() * 100.0)
        if not agreement_df.empty
        else 0.0
    )

    details_path = "reports/dor_real_sightings_model_comparison.csv"
    summary_path = "reports/dor_real_sightings_model_comparison_summary.csv"
    output_df.to_csv(details_path, index=False)
    comparison_summary.to_csv(summary_path, index=False)

    print("\n=== Real Sightings: JellyfishNet vs Baseline ===")
    print(comparison_summary.to_string(index=False))
    print(f"\nAgreement rate (both models same Yes/No): {agreement_rate:.1f}%")
    print(f"Details CSV: {details_path}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
