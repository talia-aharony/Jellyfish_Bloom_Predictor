#!/usr/bin/env python3
"""
Example: Making Predictions with the Jellyfish Forecasting System

This script demonstrates how to:
1. Load data
2. Load trained models
3. Make predictions
4. Compare models
"""

import os
import sys
import argparse

if __package__ in (None, ""):
    ROOT = os.path.dirname(os.path.dirname(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from jellyfish.predictor import JellyfishPredictor
else:
    from .predictor import JellyfishPredictor
import pandas as pd
import numpy as np
from datetime import date, timedelta


def get_user_inputs(metadata):
    """Get beach and days-ahead inputs from user"""
    
    # Display available beaches
    unique_beaches = metadata.drop_duplicates(subset=['beach_id']).sort_values('beach_id')
    
    print("\n" + "=" * 80)
    print("AVAILABLE BEACHES")
    print("=" * 80)
    print(f"{'Beach ID':<12} {'Beach Name':<50}")
    print("-" * 80)
    
    for idx, row in unique_beaches.iterrows():
        beach_id_display = int(row['beach_id'])
        beach_name_display = str(row['beach_name'])
        print(f"{beach_id_display:<12} {beach_name_display:<50}")
    
    print()
    
    # Get beach ID from user
    while True:
        try:
            beach_id = int(input("Enter Beach ID (1-20): "))
            if beach_id in unique_beaches['beach_id'].values:
                break
            else:
                print(f"❌ Beach ID {beach_id} not found. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get days ahead from user
    while True:
        try:
            days_ahead = int(input("Enter days ahead to forecast (1-365): "))
            if 1 <= days_ahead <= 365:
                break
            else:
                print("❌ Please enter a number between 1 and 365.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    return beach_id, days_ahead


def main(days_ahead=None, beach_id=None, lookback_days=7):
    """Main prediction example"""
    
    print("=" * 80)
    print("JELLYFISH FORECASTING - PREDICTION EXAMPLE")
    print("=" * 80)
    print()
    
    # =========================================================================
    # STEP 1: Initialize Predictor
    # =========================================================================
    
    print("STEP 1: Initialize Predictor")
    print("-" * 80)
    predictor = JellyfishPredictor(device='cpu')
    print()
    
    # =========================================================================
    # STEP 2: Load Data Cache
    # =========================================================================
    
    print("STEP 2: Load Data Cache")
    print("-" * 80)
    predictor.load_data_cache(lookback_days=lookback_days, forecast_days=1)
    print()
    
    # =========================================================================
    # STEP 3: Load Trained Models
    # =========================================================================
    
    print("STEP 3: Load Trained Models")
    print("-" * 80)
    
    models_to_load = [
        ('Baseline', 'baseline_model.pth', None),
        ('Feedforward', 'feedforward_model.pth', None),
        ('LSTM', 'lstm_model.pth', None),
        ('GRU', 'gru_model.pth', None),
        ('Conv1D', 'conv1d_model.pth', None),
        ('Hybrid', 'hybrid_model.pth', None),
    ]
    
    loaded_models = []
    for model_name, model_path, input_dim in models_to_load:
        try:
            if input_dim:
                predictor.load_model(model_name, model_path, input_dim=input_dim)
            else:
                predictor.load_model(model_name, model_path)
            loaded_models.append(model_name)
        except FileNotFoundError:
            print(f"⚠ Could not load {model_name} - {model_path} not found")
            print(f"  (Run 'python scripts/train.py' first to train models)")
    
    print(f"\n✓ Loaded {len(loaded_models)} models: {', '.join(loaded_models)}")
    print()
    
    if not loaded_models:
        print("ERROR: No models loaded. Please train models first:")
        print("  python scripts/train.py")
        return
    
    # =========================================================================
    # STEP 4: Get Sample Data to Predict
    # =========================================================================
    
    print("STEP 4: Get User Input")
    print("-" * 80)
    
    metadata = predictor.data_cache['metadata']
    
    # Get user inputs if not provided via command line
    if beach_id is None or days_ahead is None:
        beach_id, days_ahead = get_user_inputs(metadata)
    
    beach_id = int(beach_id)
    days_ahead = int(days_ahead)
    
    if days_ahead < 0:
        raise ValueError("days_ahead must be non-negative")
    
    beach_matches = metadata[metadata['beach_id'] == beach_id]
    if beach_matches.empty:
        print(f"❌ ERROR: beach_id {beach_id} not found in dataset metadata")
        return
    
    beach_name = str(beach_matches.iloc[0]['beach_name'])

    selected_date = date.today() + timedelta(days=days_ahead)
    print(f"\n✓ Selected Beach {beach_id}: {beach_name}")
    print(f"✓ Forecast date: {selected_date} ({days_ahead} days ahead)")
    print()
    
    # =========================================================================
    # STEP 5: Make Single Predictions
    # =========================================================================
    
    print("STEP 5: Single Predictions")
    print("-" * 80)
    forecast_date = selected_date
    
    print(f"\nPredicting for Beach {beach_id} ({beach_name}) on {forecast_date}:")
    print()
    
    for model_name in loaded_models[:3]:  # Show first 3 models
        result = predictor.predict_days_ahead(
            beach_id=beach_id,
            days_ahead=days_ahead,
            model_name=model_name,
        )
        
        if 'error' in result:
            print(f"  {model_name:15s}: {result['error']}")
        else:
            print(f"  {model_name:15s}: {result['percentage']:6.1f}% "
                  f"({result['prediction']:3s}) - Confidence: {result['confidence']}")
            if result.get('extrapolated'):
                print(f"{'':19s}  ↳ Extrapolated from latest known date: {result.get('extrapolated_from_date')}")
    
    print()
    
    # =========================================================================
    # STEP 6: Hybrid Model Comparison View
    # =========================================================================
    
    print("STEP 6: Hybrid Model Details")
    print("-" * 80)
    
    predictor.compare_predictions(beach_id=beach_id, forecast_date=forecast_date)
    
    # =========================================================================
    # STEP 7: Final Prediction Table
    # =========================================================================
    
    print("STEP 7: Prediction Summary")
    print("-" * 80)
    
    # Create list of beach-date combinations to predict
    predictions_list = []
    predictions_list.append((beach_id, selected_date))
    
    print(f"\nGenerating summary for {len(predictions_list)} beach-date combination:")
    print()
    
    results = predictor.predict_multiple(
        predictions_list,
        model_name=loaded_models[0]  # Use first loaded model (usually Baseline)
    )
    
    # Display results as table
    df_results = []
    for result in results:
        if 'error' not in result:
            df_results.append({
                'Beach ID': int(result['beach_id']),
                'Beach Name': result['beach_name'][:15],
                'Date': str(result['forecast_date']),
                'Probability': f"{result['percentage']:.1f}%",
                'Prediction': result['prediction'],
                'Confidence': result['confidence']
            })
    
    if df_results:
        df = pd.DataFrame(df_results)
        print(df.to_string(index=False))
    print()
    
    # =========================================================================
    # STEP 8: Ensemble Predictions
    # =========================================================================
    
    if len(loaded_models) > 1:
        print("STEP 8: Ensemble Predictions")
        print("-" * 80)
        
        print(f"\nGetting predictions from all {len(loaded_models)} models:")
        print()
        
        all_results = predictor.predict_all_models(
            beach_id=beach_id,
            forecast_date=forecast_date
        )
        
        # Calculate ensemble
        probabilities = []
        model_predictions = []
        
        print(f"{'Model':<20} {'Probability':<15} {'Percentage':<15}")
        print("-" * 50)
        
        for model_name in sorted(all_results.keys()):
            result = all_results[model_name]
            if 'error' not in result:
                prob = result['probability']
                pct = result['percentage']
                probabilities.append(prob)
                model_predictions.append(pct)
                pct_display = f"{pct:.2f}%"
                print(f"{model_name:<20} {prob:<15.4f} {pct_display:<15}")
       
        if probabilities:
            ensemble_prob = np.mean(probabilities)
            ensemble_pct = ensemble_prob * 100
            ensemble_pred = 'Yes' if ensemble_prob > 0.5 else 'No'
            ensemble_pct_display = f"{ensemble_pct:.2f}%"
            yes_votes = sum(p > 0.5 for p in probabilities)
            no_votes = len(probabilities) - yes_votes
            
            print("-" * 50)
            print(f"{'ENSEMBLE':<20} {ensemble_prob:<15.4f} {ensemble_pct_display:<15}")
            print()
            print(f"Ensemble Prediction: {ensemble_pred}")
            print(f"  Average probability: {ensemble_pct:.2f}%")
            print(f"  Number of models: {len(probabilities)}")
            print(f"  Votes: {yes_votes}/{len(probabilities)} predict 'Yes', {no_votes}/{len(probabilities)} predict 'No'")
        
        print()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"✓ Successfully demonstrated prediction system")
    print(f"✓ Loaded {len(loaded_models)} model(s)")
    print(f"✓ Made predictions for {len(predictions_list)} beach-date combination(s)")
    print()
    print("Next steps:")
    print("  1. Modify this script to predict for your dates of interest")
    print("  2. Use predictor.predict_for_beach_date() in your own code")
    print("  3. Load different models and compare predictions")
    print()
    print("=" * 80)

   



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run jellyfish prediction example")
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=None,
        help="Forecast horizon in days from today (interactive if not provided)",
    )
    parser.add_argument(
        "--beach-id",
        type=int,
        default=None,
        help="Beach ID to target for prediction (interactive if not provided)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Historical input window length in days (default: 7)",
    )
    args = parser.parse_args()
    main(days_ahead=args.days_ahead, beach_id=args.beach_id, lookback_days=args.lookback_days)
