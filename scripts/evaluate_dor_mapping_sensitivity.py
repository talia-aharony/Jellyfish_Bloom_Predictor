import itertools

import pandas as pd

from jellyfish.predictor import JellyfishPredictor
from scripts.evaluate_dor_reports import parse_dor_reports, get_beach_mapping


def evaluate_scenario(name, mapping, reports_df, predictor):
    scenario_df = reports_df.copy()
    scenario_df["model_beach_id"] = scenario_df["meduzot_beach_id"].map(mapping)

    predictions = []
    for _, row in scenario_df.iterrows():
        meduzot_beach_id = int(row["meduzot_beach_id"])
        model_beach_id = row["model_beach_id"]

        if pd.isna(model_beach_id):
            predictions.append(
                {
                    "scenario": name,
                    "observation_id": int(row["observation_id"]),
                    "meduzot_beach_id": meduzot_beach_id,
                    "model_beach_id": None,
                    "pred_yes_no": None,
                    "pred_percentage": None,
                    "pred_error": "No model beach mapping",
                }
            )
            continue

        result = predictor.predict_for_beach_date(
            int(model_beach_id), row["report_date"], "JellyfishNet"
        )
        if "error" in result:
            predictions.append(
                {
                    "scenario": name,
                    "observation_id": int(row["observation_id"]),
                    "meduzot_beach_id": meduzot_beach_id,
                    "model_beach_id": int(model_beach_id),
                    "pred_yes_no": None,
                    "pred_percentage": None,
                    "pred_error": result["error"],
                }
            )
        else:
            predictions.append(
                {
                    "scenario": name,
                    "observation_id": int(row["observation_id"]),
                    "meduzot_beach_id": meduzot_beach_id,
                    "model_beach_id": int(model_beach_id),
                    "pred_yes_no": result["prediction"],
                    "pred_percentage": float(result["percentage"]),
                    "pred_error": None,
                }
            )

    pred_df = pd.DataFrame(predictions)
    summary = {
        "scenario": name,
        "predictions": int(pred_df["pred_yes_no"].notna().sum()),
        "pred_yes": int((pred_df["pred_yes_no"] == "Yes").sum()),
        "pred_no": int((pred_df["pred_yes_no"] == "No").sum()),
        "missing": int(pred_df["pred_yes_no"].isna().sum()),
        "yes_rate_pct": float((pred_df["pred_yes_no"] == "Yes").mean() * 100.0),
    }
    return pred_df, summary


def main():
    reports_df = parse_dor_reports()

    base_mapping = get_beach_mapping()

    # Ambiguous regional labels to test:
    # 24 nמל חיפה והחוף השקט  -> {3,4,5}
    # 23 בת גלים              -> {3,4,5}
    # 21 צפון חוף הכרמל       -> {5,6,7}
    # 10 תל אביב צפון         -> {13,14,15}
    options = {
        24: [3, 4, 5],
        23: [3, 4, 5],
        21: [5, 6, 7],
        10: [13, 14, 15],
    }

    predictor = JellyfishPredictor(device="cpu")
    predictor.load_data_cache(
        lookback_days=24,
        forecast_days=1,
        weather_csv_path=None,
        include_live_xml=False,
    )
    predictor.load_model("JellyfishNet", "models/jellyfishnet_model.pth")

    scenario_rows = []

    # Baseline (current mapping)
    base_pred, base_summary = evaluate_scenario("baseline_mapping", base_mapping, reports_df, predictor)
    scenario_rows.append(base_summary)

    # Cartesian combinations of ambiguous mapping choices
    keys = sorted(options.keys())
    all_combos = list(itertools.product(*[options[k] for k in keys]))

    all_pred_frames = [base_pred]
    for combo in all_combos:
        mapping = dict(base_mapping)
        parts = []
        for key, mapped in zip(keys, combo):
            mapping[key] = mapped
            parts.append(f"{key}->{mapped}")
        scenario_name = "alt_" + "_".join(parts)

        pred_df, summary = evaluate_scenario(scenario_name, mapping, reports_df, predictor)
        all_pred_frames.append(pred_df)
        scenario_rows.append(summary)

    summary_df = pd.DataFrame(scenario_rows)
    summary_df = summary_df.sort_values(["pred_yes", "yes_rate_pct"], ascending=[False, False])

    all_pred_df = pd.concat(all_pred_frames, ignore_index=True)

    base_yes = set(base_pred.loc[base_pred["pred_yes_no"] == "Yes", "observation_id"].tolist())
    deltas = []
    for scenario in summary_df["scenario"]:
        if scenario == "baseline_mapping":
            continue
        scen_df = all_pred_df[all_pred_df["scenario"] == scenario]
        scen_yes = set(scen_df.loc[scen_df["pred_yes_no"] == "Yes", "observation_id"].tolist())
        flipped_to_yes = len(scen_yes - base_yes)
        flipped_to_no = len(base_yes - scen_yes)
        deltas.append(
            {
                "scenario": scenario,
                "flipped_to_yes_vs_baseline": flipped_to_yes,
                "flipped_to_no_vs_baseline": flipped_to_no,
                "total_changed_vs_baseline": flipped_to_yes + flipped_to_no,
            }
        )

    delta_df = pd.DataFrame(deltas).sort_values("total_changed_vs_baseline", ascending=False)

    summary_path = "reports/final/dor/sensitivity/summary.csv"
    delta_path = "reports/final/dor/sensitivity/changes.csv"
    all_pred_path = "reports/final/dor/sensitivity/all_predictions.csv"

    summary_df.to_csv(summary_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    all_pred_df.to_csv(all_pred_path, index=False)

    print("=== Mapping Sensitivity Summary (top 10 by predicted Yes) ===")
    print(summary_df.head(10).to_string(index=False))

    print("\n=== Most different scenarios vs baseline (top 10) ===")
    print(delta_df.head(10).to_string(index=False))

    print("\nSaved:")
    print(summary_path)
    print(delta_path)
    print(all_pred_path)


if __name__ == "__main__":
    main()
