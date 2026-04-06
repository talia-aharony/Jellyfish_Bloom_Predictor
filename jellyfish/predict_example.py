#!/usr/bin/env python3
"""Example: make jellyfish predictions with Baseline and JellyfishNet."""

import argparse
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    ROOT = os.path.dirname(os.path.dirname(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from jellyfish.predictor import JellyfishPredictor
    from jellyfish.settings import DEFAULT_LOOKBACK_DAYS, DEFAULT_WEATHER_CSV_PATH
    from jellyfish.terminal_format import rule, section
else:
    from .predictor import JellyfishPredictor
    from .settings import DEFAULT_LOOKBACK_DAYS, DEFAULT_WEATHER_CSV_PATH
    from .terminal_format import rule, section


DEFAULT_BASELINE_MODEL_PATH = "baseline_model.pth"
DEFAULT_JELLYFISHNET_MODEL_PATH = "model_runs/st017_jellynet_e100/jellyfishnet_model.pth"


def get_user_inputs(metadata):
    """Get beach and days-ahead inputs from user."""
    unique_beaches = metadata.drop_duplicates(subset=["beach_id"]).sort_values("beach_id")

    section("AVAILABLE BEACHES")
    header = f"{'Beach ID':<12} {'Beach Name':<50}"
    print(header)
    print(rule(header, fill="-"))
    for _, row in unique_beaches.iterrows():
        print(f"{int(row['beach_id']):<12} {str(row['beach_name']):<50}")
    print()

    while True:
        try:
            beach_id = int(input("Enter Beach ID (1-20): "))
            if beach_id in unique_beaches["beach_id"].values:
                break
            print(f"❌ Beach ID {beach_id} not found. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

    while True:
        try:
            days_ahead = int(input("Enter days ahead to forecast (1-365): "))
            if 1 <= days_ahead <= 365:
                break
            print("❌ Please enter a number between 1 and 365.")
        except ValueError:
            print("❌ Please enter a valid number.")

    return beach_id, days_ahead


def main(
    days_ahead=None,
    beach_id=None,
    lookback_days=DEFAULT_LOOKBACK_DAYS,
    weather_csv_path=DEFAULT_WEATHER_CSV_PATH,
    include_live_xml=True,
    fast_preview=False,
    preview_samples=1500,
    baseline_model_path=DEFAULT_BASELINE_MODEL_PATH,
    jellyfishnet_model_path=DEFAULT_JELLYFISHNET_MODEL_PATH,
):
    """Main prediction example."""
    section("JELLYFISH FORECASTING - PREDICTION EXAMPLE")

    section("STEP 1: Initialize Predictor", fill="-")
    predictor = JellyfishPredictor(device="cpu")
    print()

    section("STEP 2: Load Data Cache", fill="-")
    effective_include_live_xml = include_live_xml
    max_cache_samples = None
    if fast_preview:
        effective_include_live_xml = False
        max_cache_samples = int(preview_samples)
        print(
            f"⚡ Fast preview mode enabled: loading integrated cache "
            f"with up to {max_cache_samples} sequences"
        )

    predictor.load_data_cache(
        lookback_days=lookback_days,
        forecast_days=1,
        weather_csv_path=weather_csv_path,
        include_live_xml=effective_include_live_xml,
        max_cache_samples=max_cache_samples,
    )
    print()

    section("STEP 3: Load Trained Models", fill="-")
    models_to_load = [
        ("Baseline", baseline_model_path),
        ("JellyfishNet", jellyfishnet_model_path),
    ]

    loaded_models = []
    for model_name, model_path in models_to_load:
        try:
            predictor.load_model(model_name, model_path)
            loaded_models.append(model_name)
        except FileNotFoundError:
            print(f"⚠ Could not load {model_name} - {model_path} not found")
            print("  (Run 'python scripts/train.py' first to train models)")

    print(f"\n✓ Loaded {len(loaded_models)} models: {', '.join(loaded_models)}")
    print()

    if not loaded_models:
        print("ERROR: No models loaded. Please train models first:")
        print("  python scripts/train.py")
        return

    section("STEP 4: Get User Input", fill="-")
    metadata = predictor.data_cache["metadata"]
    if beach_id is None or days_ahead is None:
        beach_id, days_ahead = get_user_inputs(metadata)

    beach_id = int(beach_id)
    days_ahead = int(days_ahead)
    if days_ahead < 0:
        raise ValueError("days_ahead must be non-negative")

    beach_matches = metadata[metadata["beach_id"] == beach_id]
    if beach_matches.empty:
        print(f"❌ ERROR: beach_id {beach_id} not found in dataset metadata")
        return

    beach_name = str(beach_matches.iloc[0]["beach_name"])
    selected_date = date.today() + timedelta(days=days_ahead)
    print(f"\n✓ Selected Beach {beach_id}: {beach_name}")
    print(f"✓ Forecast date: {selected_date} ({days_ahead} days ahead)")
    print()

    section("STEP 5: Single Predictions", fill="-")
    forecast_date = selected_date
    print(f"\nPredicting for Beach {beach_id} ({beach_name}) on {forecast_date}:")
    print()

    successful_models = []
    for model_name in loaded_models:
        try:
            result = predictor.predict_days_ahead(
                beach_id=beach_id,
                days_ahead=days_ahead,
                model_name=model_name,
            )
        except Exception as exc:
            print(f"  {model_name:15s}: ERROR - {exc}")
            continue

        if "error" in result:
            print(f"  {model_name:15s}: {result['error']}")
        else:
            successful_models.append(model_name)
            print(
                f"  {model_name:15s}: {result['percentage']:6.1f}% "
                f"({result['prediction']:3s}) - Confidence: {result['confidence']}"
            )
            if result.get("extrapolated"):
                print(
                    f"{'':19s}  ↳ Extrapolated from latest known date: "
                    f"{result.get('extrapolated_from_date')}"
                )
    print()

    section("STEP 6: Model Comparison", fill="-")
    if successful_models:
        predictor.compare_predictions(beach_id=beach_id, forecast_date=forecast_date)
    else:
        print("No compatible model predictions available for comparison.")

    section("STEP 7: Prediction Summary", fill="-")
    predictions_list = [(beach_id, selected_date)]
    print(f"\nGenerating summary for {len(predictions_list)} beach-date combination:")
    print()

    rows = []
    if successful_models:
        summary_model = successful_models[0]
        results = predictor.predict_multiple(predictions_list, model_name=summary_model)
        for result in results:
            if "error" not in result:
                rows.append(
                    {
                        "Beach ID": int(result["beach_id"]),
                        "Beach Name": result["beach_name"][:15],
                        "Date": str(result["forecast_date"]),
                        "Probability": f"{result['percentage']:.1f}%",
                        "Prediction": result["prediction"],
                        "Confidence": result["confidence"],
                    }
                )
    else:
        print("No compatible model available for summary table.")

    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    print()

    if len(loaded_models) > 1:
        section("STEP 8: Ensemble Predictions", fill="-")
        print(f"\nGetting predictions from all {len(loaded_models)} models:")
        print()

        all_results = predictor.predict_all_models(beach_id=beach_id, forecast_date=forecast_date)
        header = f"{'Model':<20} {'Probability':<15} {'Percentage':<15}"
        print(header)
        print(rule(header, fill="-"))

        probabilities = []
        for model_name in sorted(all_results.keys()):
            result = all_results[model_name]
            if "error" not in result:
                prob = result["probability"]
                pct = result["percentage"]
                probabilities.append(prob)
                print(f"{model_name:<20} {prob:<15.4f} {pct:.2f}%")

        if probabilities:
            ensemble_prob = float(np.mean(probabilities))
            ensemble_pct = ensemble_prob * 100
            ensemble_pred = "Yes" if ensemble_prob > 0.5 else "No"
            yes_votes = sum(p > 0.5 for p in probabilities)
            no_votes = len(probabilities) - yes_votes

            print(rule(header, fill="-"))
            print(f"{'ENSEMBLE':<20} {ensemble_prob:<15.4f} {ensemble_pct:.2f}%")
            print()
            print(f"Ensemble Prediction: {ensemble_pred}")
            print(f"  Average probability: {ensemble_pct:.2f}%")
            print(f"  Number of models: {len(probabilities)}")
            print(
                f"  Votes: {yes_votes}/{len(probabilities)} predict 'Yes', "
                f"{no_votes}/{len(probabilities)} predict 'No'"
            )
        print()

    section("SUMMARY")
    print("✓ Successfully demonstrated prediction system")
    print(f"✓ Loaded {len(loaded_models)} model(s)")
    print(f"✓ Made predictions for {len(predictions_list)} beach-date combination(s)")
    print()
    print("Next steps:")
    print("  1. Modify this script to predict for your dates of interest")
    print("  2. Use predictor.predict_for_beach_date() in your own code")
    print("  3. Load different models and compare predictions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run jellyfish prediction example")
    parser.add_argument("--days-ahead", type=int, default=None)
    parser.add_argument("--beach-id", type=int, default=None)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Historical input window length in days (default: {DEFAULT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--weather-csv-path",
        type=str,
        default=DEFAULT_WEATHER_CSV_PATH,
        help="Path to IMS weather CSV for integrated data loading",
    )
    parser.add_argument(
        "--disable-live-xml",
        action="store_true",
        help="Disable live RSS XML enrichment in integrated mode",
    )
    parser.add_argument(
        "--fast-preview",
        action="store_true",
        help="Run a lightweight demo path with smaller integrated cache",
    )
    parser.add_argument(
        "--preview-samples",
        type=int,
        default=1500,
        help="Max sequences to cache when --fast-preview is enabled (default: 1500)",
    )
    parser.add_argument(
        "--baseline-model-path",
        type=str,
        default=DEFAULT_BASELINE_MODEL_PATH,
        help=f"Path to baseline checkpoint (default: {DEFAULT_BASELINE_MODEL_PATH})",
    )
    parser.add_argument(
        "--jellyfishnet-model-path",
        type=str,
        default=DEFAULT_JELLYFISHNET_MODEL_PATH,
        help=f"Path to JellyfishNet checkpoint (default: {DEFAULT_JELLYFISHNET_MODEL_PATH})",
    )
    args = parser.parse_args()
    main(
        days_ahead=args.days_ahead,
        beach_id=args.beach_id,
        lookback_days=args.lookback_days,
        weather_csv_path=args.weather_csv_path,
        include_live_xml=not args.disable_live_xml,
        fast_preview=args.fast_preview,
        preview_samples=args.preview_samples,
        baseline_model_path=args.baseline_model_path,
        jellyfishnet_model_path=args.jellyfishnet_model_path,
    )
