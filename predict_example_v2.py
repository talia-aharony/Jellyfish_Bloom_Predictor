#!/usr/bin/env python3
"""
Clean Example: Baseline vs Your Model

Compare simple logistic regression against your hybrid CNN+LSTM model.
Shows which model performs better across different beaches and dates.
"""

from predictor_v2 import JellyfishPredictor
import pandas as pd


def main():
    print("=" * 80)
    print("BASELINE vs YOUR MODEL - JELLYFISH PREDICTION")
    print("=" * 80)
    print()
    
    # Initialize
    predictor = JellyfishPredictor(device='cpu')
    
    # Load data
    predictor.load_data_cache(lookback_days=7, forecast_days=1)
    
    # Get sample beaches/dates from data
    metadata = predictor.data_cache['metadata']
    sample_predictions = [
        (int(row['beach_id']), row['forecast_date']) 
        for _, row in metadata.drop_duplicates(subset=['beach_id']).head(5).iterrows()
    ]
    
    # Load models
    print("\nLoading Models:")
    print("-" * 80)
    try:
        predictor.load_baseline_model('baseline_model.pth')
    except FileNotFoundError:
        print("❌ baseline_model.pth not found - run train.py first")
        return
    
    try:
        predictor.load_your_model('hybrid_model.pth')
    except FileNotFoundError:
        print("❌ hybrid_model.pth not found - run train.py first")
        return
    
    print()
    
    # =========================================================================
    # EXAMPLE 1: Single Prediction with Details
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("EXAMPLE 1: SINGLE PREDICTION - DETAILED COMPARISON")
    print("=" * 80)
    
    beach_id = int(sample_predictions[0][0])
    forecast_date = sample_predictions[0][1]
    
    result = predictor.predict_for_beach_date(
        beach_id=beach_id,
        forecast_date=forecast_date,
        show_details=True
    )
    
    # =========================================================================
    # EXAMPLE 2: Model Agreement Analysis
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: DO THE MODELS AGREE?")
    print("=" * 80)
    print()
    
    agreement = predictor.get_model_agreement(beach_id, forecast_date)
    
    print(f"Beach ID: {beach_id}")
    print(f"Date: {forecast_date}")
    print()
    print(f"Baseline prediction:    {agreement['baseline']}")
    print(f"Your model prediction:  {agreement['your_model']}")
    print(f"Actual outcome:         {agreement['actual']}")
    print()
    print(f"Models agree:           {'✓ YES' if agreement['agree'] else '✗ NO'}")
    print(f"Both correct:           {'✓ YES' if agreement['both_correct'] else '✗ NO'}")
    print(f"Only yours correct:     {'✓ YES' if agreement['only_yours_correct'] else '✗ NO'}")
    print(f"Only baseline correct:  {'✓ YES' if agreement['only_baseline_correct'] else '✗ NO'}")
    print(f"Both wrong:             {'✓ YES' if agreement['both_wrong'] else '✗ NO'}")
    
    # =========================================================================
    # EXAMPLE 3: Multiple Predictions Comparison
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: COMPARE ACROSS MULTIPLE PREDICTIONS")
    print("=" * 80)
    print()
    
    df = predictor.compare_multiple_predictions(sample_predictions[:10])
    print(df.to_string(index=False))
    
    # =========================================================================
    # EXAMPLE 4: Which Model Wins?
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("EXAMPLE 4: SCORECARD - YOUR MODEL vs BASELINE")
    print("=" * 80)
    print()
    
    your_model_wins = 0
    baseline_wins = 0
    both_correct = 0
    both_wrong = 0
    
    for beach_id, forecast_date in sample_predictions[:20]:
        agreement = predictor.get_model_agreement(beach_id, forecast_date)
        
        if agreement:
            if agreement['both_correct']:
                both_correct += 1
                your_model_wins += 1
                baseline_wins += 1
            elif agreement['only_yours_correct']:
                your_model_wins += 1
            elif agreement['only_baseline_correct']:
                baseline_wins += 1
            elif agreement['both_wrong']:
                both_wrong += 1
    
    evaluated_predictions = sample_predictions[:20]
    total = len(evaluated_predictions)
    print(f"Total predictions tested: {total}")
    print()
    print(f"Your Model:     {your_model_wins:2d} correct ({(your_model_wins/total*100) if total else 0:5.1f}%)")
    print(f"Baseline:       {baseline_wins:2d} correct ({(baseline_wins/total*100) if total else 0:5.1f}%)")
    print()
    print(f"Both correct:   {both_correct:2d}")
    print(f"Both wrong:     {both_wrong:2d}")
    
    if your_model_wins > baseline_wins:
        improvement = your_model_wins - baseline_wins
        print(f"\n✓ Your Model is better by {improvement} predictions!")
    elif baseline_wins > your_model_wins:
        print(f"\n⚠ Baseline is better - your model needs improvement")
    else:
        print(f"\n= Models are equivalent")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
✓ Baseline Model:
  - Simple logistic regression
    - Uses engineered features (110 inputs)
  - Fast, interpretable, but limited

✓ Your Model:
  - Hybrid CNN + LSTM
  - Processes temporal sequences (7 days × 11 features)
  - More complex but captures temporal patterns

Goal: Show your model outperforms the baseline!

Key Metrics:
  - Accuracy on unseen data
  - Agreement with baseline
  - Correctness when they disagree

Next Steps:
  1. Train both models properly
  2. Run this comparison on test set
  3. Analyze where your model wins
  4. Document improvements in your paper
    """)


if __name__ == '__main__':
    main()
