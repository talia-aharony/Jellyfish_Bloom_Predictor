#!/usr/bin/env python3
"""
Model Evaluation Example

Shows how to evaluate Baseline vs Your Model with comprehensive metrics:
- Accuracy, Precision, Recall, F1-Score
- Sensitivity, Specificity
- Confusion Matrix
- ROC-AUC
- Head-to-head comparison
"""

from predictor_v2 import JellyfishPredictor
from evaluator import ModelEvaluator
import pandas as pd
import numpy as np


def main():
    print("=" * 100)
    print("MODEL EVALUATION - BASELINE vs YOUR MODEL")
    print("=" * 100)
    print()
    
    # =========================================================================
    # STEP 1: Initialize
    # =========================================================================
    
    print("STEP 1: Initialize Predictor and Evaluator")
    print("-" * 100)
    
    predictor = JellyfishPredictor(device='cpu')
    predictor.load_data_cache(lookback_days=7, forecast_days=1)
    
    # Load models
    try:
        predictor.load_baseline_model('baseline_model.pth')
        predictor.load_your_model('hybrid_model.pth')
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run 'python train.py' first to train the models")
        return
    
    print()
    
    # =========================================================================
    # STEP 2: Create Test Set
    # =========================================================================
    
    print("STEP 2: Create Test Set")
    print("-" * 100)
    
    metadata = predictor.data_cache['metadata']
    
    # Get diverse test set (mix of beaches and dates)
    # For evaluation, use last 20% of data
    test_size = int(len(metadata) * 0.2)
    test_indices = np.random.choice(len(metadata), test_size, replace=False)
    
    test_set = [
        (int(row['beach_id']), row['forecast_date'])
        for _, row in metadata.iloc[test_indices].iterrows()
    ]
    
    print(f"Total samples in dataset: {len(metadata)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Test set will be {len(test_set)/len(metadata)*100:.1f}% of data")
    print()
    
    # =========================================================================
    # STEP 3: Evaluate
    # =========================================================================
    
    print("STEP 3: Evaluating Models")
    print("-" * 100)
    
    evaluator = ModelEvaluator(predictor)
    results = evaluator.evaluate_on_dataset(test_set, verbose=True)
    
    print()
    
    # =========================================================================
    # STEP 4: View Metrics as Table
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("METRICS TABLE (For Easy Comparison)")
    print("=" * 100)
    print()
    
    df = evaluator.get_metrics_dataframe()
    print(df.to_string(index=False))
    
    print()
    
    # =========================================================================
    # STEP 5: Confusion Matrix
    # =========================================================================
    
    print("\n" + "=" * 100)
    print("CONFUSION MATRIX DETAILS")
    print("=" * 100)
    print()
    
    cm = evaluator.get_confusion_matrix_summary()
    
    print("Baseline Confusion Matrix:")
    print(f"              Predicted Negative | Predicted Positive")
    print(f"Actual Neg:   {cm['baseline']['tn']:18d} | {cm['baseline']['fp']:18d}")
    print(f"Actual Pos:   {cm['baseline']['fn']:18d} | {cm['baseline']['tp']:18d}")
    
    print()
    
    print("Your Model Confusion Matrix:")
    print(f"              Predicted Negative | Predicted Positive")
    print(f"Actual Neg:   {cm['your_model']['tn']:18d} | {cm['your_model']['fp']:18d}")
    print(f"Actual Pos:   {cm['your_model']['fn']:18d} | {cm['your_model']['tp']:18d}")
    
    print()
    
    # =========================================================================
    # STEP 6: Export Results
    # =========================================================================
    
    print("=" * 100)
    print("EXPORTING RESULTS")
    print("=" * 100)
    print()
    
    # Export metrics to CSV
    evaluator.export_metrics_to_csv('model_metrics.csv')
    
    # Export full report
    evaluator.export_report('model_evaluation_report.txt')
    
    print()
    
    # =========================================================================
    # STEP 7: Summary
    # =========================================================================
    
    print("=" * 100)
    print("SUMMARY FOR YOUR PAPER")
    print("=" * 100)
    print()
    
    comp = results['comparison']
    baseline_metrics = results['baseline']
    your_model_metrics = results['your_model']
    
    print("✓ Baseline Model:")
    print(f"  - Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"  - Precision: {baseline_metrics['precision']:.4f}")
    print(f"  - Recall: {baseline_metrics['recall']:.4f}")
    print(f"  - F1-Score: {baseline_metrics['f1']:.4f}")
    
    print()
    
    print("✓ Your Model (Hybrid CNN+LSTM):")
    print(f"  - Accuracy: {your_model_metrics['accuracy']:.4f}")
    print(f"  - Precision: {your_model_metrics['precision']:.4f}")
    print(f"  - Recall: {your_model_metrics['recall']:.4f}")
    print(f"  - F1-Score: {your_model_metrics['f1']:.4f}")
    
    print()
    
    print("✓ Improvement:")
    print(f"  - Accuracy improvement: {comp['accuracy_improvement_pct']:+.2f}%")
    print(f"  - Your model wins: {comp['your_model_wins']} predictions")
    print(f"  - Baseline wins: {comp['baseline_wins']} predictions")
    
    print()
    print("✓ Files exported:")
    print(f"  - model_metrics.csv (for tables in paper)")
    print(f"  - model_evaluation_report.txt (full report)")
    
    print()


if __name__ == '__main__':
    import numpy as np
    main()
