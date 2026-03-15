"""
Model Evaluation Module - Comprehensive Metrics

Computes and compares metrics for Baseline vs Your Model:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Confusion Matrix
- Agreement analysis
- Statistical comparison
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from datetime import datetime


class ModelEvaluator:
    """Evaluate and compare Baseline vs Your Model"""
    
    def __init__(self, predictor):
        """
        Initialize evaluator
        
        Args:
            predictor: JellyfishPredictor instance with models loaded
        """
        self.predictor = predictor
        self.baseline_metrics = None
        self.your_model_metrics = None
        self.comparison = None
    
    def evaluate_on_dataset(self, test_predictions_list, verbose=True):
        """
        Evaluate both models on a list of predictions
        
        Args:
            test_predictions_list: List of (beach_id, forecast_date) tuples
            verbose: Print results
        
        Returns:
            Dict with metrics for both models
        """
        
        baseline_preds = []
        baseline_probs = []
        your_model_preds = []
        your_model_probs = []
        actuals = []
        
        print(f"Evaluating on {len(test_predictions_list)} predictions...")
        
        for beach_id, forecast_date in test_predictions_list:
            result = self.predictor.predict_for_beach_date(beach_id, forecast_date)
            
            if 'error' not in result:
                # Get predictions and probabilities
                baseline_preds.append(1 if result['baseline']['prediction'] == 'Yes' else 0)
                baseline_probs.append(result['baseline']['probability'])
                
                your_model_preds.append(1 if result['your_model']['prediction'] == 'Yes' else 0)
                your_model_probs.append(result['your_model']['probability'])
                
                actuals.append(1 if result['actual'] == 'Yes' else 0)
        
        # Convert to arrays
        baseline_preds = np.array(baseline_preds)
        baseline_probs = np.array(baseline_probs)
        your_model_preds = np.array(your_model_preds)
        your_model_probs = np.array(your_model_probs)
        actuals = np.array(actuals)
        
        # Compute metrics
        self.baseline_metrics = self._compute_metrics(
            actuals, baseline_preds, baseline_probs, 'Baseline'
        )
        self.your_model_metrics = self._compute_metrics(
            actuals, your_model_preds, your_model_probs, 'Your Model'
        )
        
        # Compute comparison
        self.comparison = self._compute_comparison(
            baseline_preds, your_model_preds, actuals
        )
        
        if verbose:
            self._print_metrics()
        
        return {
            'baseline': self.baseline_metrics,
            'your_model': self.your_model_metrics,
            'comparison': self.comparison
        }
    
    def _compute_metrics(self, y_true, y_pred, y_proba, model_name):
        """Compute all metrics for a single model"""
        
        metrics = {
            'model_name': model_name,
            'n_samples': len(y_true),
            'n_positive': np.sum(y_true),
            'n_negative': len(y_true) - np.sum(y_true),
        }
        
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = None
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['tn'] = tn
        metrics['fp'] = fp
        metrics['fn'] = fn
        metrics['tp'] = tp
        
        # Sensitivity and Specificity
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        return metrics
    
    def _compute_comparison(self, baseline_preds, your_model_preds, actuals):
        """Compare baseline vs your model"""
        
        comparison = {}
        
        # Agreement
        agree = (baseline_preds == your_model_preds).astype(int)
        comparison['agreement_rate'] = np.mean(agree)
        comparison['total_agreement'] = np.sum(agree)
        comparison['total_disagreement'] = len(agree) - np.sum(agree)
        
        # When they agree, are they correct?
        both_correct = np.sum((baseline_preds == actuals) & (your_model_preds == actuals) & agree)
        both_wrong = np.sum((baseline_preds != actuals) & (your_model_preds != actuals) & agree)
        
        comparison['both_correct_when_agree'] = both_correct
        comparison['both_wrong_when_agree'] = both_wrong
        
        # Wins/Losses
        only_baseline_correct = np.sum((baseline_preds == actuals) & (your_model_preds != actuals))
        only_your_model_correct = np.sum((your_model_preds == actuals) & (baseline_preds != actuals))
        
        comparison['baseline_wins'] = only_baseline_correct
        comparison['your_model_wins'] = only_your_model_correct
        comparison['net_improvement'] = only_your_model_correct - only_baseline_correct
        
        # Improvement percentage
        baseline_correct = np.sum(baseline_preds == actuals)
        your_model_correct = np.sum(your_model_preds == actuals)
        comparison['baseline_accuracy'] = baseline_correct / len(actuals)
        comparison['your_model_accuracy'] = your_model_correct / len(actuals)
        comparison['accuracy_improvement'] = comparison['your_model_accuracy'] - comparison['baseline_accuracy']
        comparison['accuracy_improvement_pct'] = comparison['accuracy_improvement'] * 100
        
        return comparison
    
    def _print_metrics(self):
        """Print formatted metrics comparison"""
        
        if self.baseline_metrics is None or self.your_model_metrics is None:
            print("No metrics computed yet. Run evaluate_on_dataset() first.")
            return
        
        print("\n" + "=" * 100)
        print("MODEL EVALUATION METRICS - BASELINE vs YOUR MODEL")
        print("=" * 100)
        print()
        
        # =====================================================================
        # CLASSIFICATION METRICS
        # =====================================================================
        
        print("CLASSIFICATION METRICS")
        print("-" * 100)
        print(f"{'Metric':<25} {'Baseline':<20} {'Your Model':<20} {'Improvement':<20}")
        print("-" * 100)
        
        metrics_to_show = [
            ('Accuracy', 'accuracy'),
            ('Precision', 'precision'),
            ('Recall', 'recall'),
            ('F1-Score', 'f1'),
            ('Balanced Accuracy', 'balanced_accuracy'),
            ('ROC-AUC', 'roc_auc'),
        ]
        
        for metric_name, metric_key in metrics_to_show:
            baseline_val = self.baseline_metrics.get(metric_key)
            your_model_val = self.your_model_metrics.get(metric_key)
            
            if baseline_val is None or your_model_val is None:
                continue
            
            improvement = your_model_val - baseline_val
            sign = "+" if improvement >= 0 else ""
            
            print(f"{metric_name:<25} {baseline_val:<20.4f} {your_model_val:<20.4f} {sign}{improvement:<19.4f}")
        
        print()
        
        # =====================================================================
        # SENSITIVITY & SPECIFICITY
        # =====================================================================
        
        print("SENSITIVITY & SPECIFICITY")
        print("-" * 100)
        print(f"{'Metric':<25} {'Baseline':<20} {'Your Model':<20} {'Improvement':<20}")
        print("-" * 100)
        
        sens_baseline = self.baseline_metrics['sensitivity']
        sens_your = self.your_model_metrics['sensitivity']
        spec_baseline = self.baseline_metrics['specificity']
        spec_your = self.your_model_metrics['specificity']
        
        print(f"{'Sensitivity (Recall)':<25} {sens_baseline:<20.4f} {sens_your:<20.4f} {sens_your-sens_baseline:+.4f}")
        print(f"{'Specificity':<25} {spec_baseline:<20.4f} {spec_your:<20.4f} {spec_your-spec_baseline:+.4f}")
        
        print()
        
        # =====================================================================
        # CONFUSION MATRIX
        # =====================================================================
        
        print("CONFUSION MATRIX")
        print("-" * 100)
        
        print("\nBaseline:")
        print(f"  True Negatives:  {self.baseline_metrics['tn']:6d}")
        print(f"  False Positives: {self.baseline_metrics['fp']:6d}")
        print(f"  False Negatives: {self.baseline_metrics['fn']:6d}")
        print(f"  True Positives:  {self.baseline_metrics['tp']:6d}")
        
        print("\nYour Model:")
        print(f"  True Negatives:  {self.your_model_metrics['tn']:6d}")
        print(f"  False Positives: {self.your_model_metrics['fp']:6d}")
        print(f"  False Negatives: {self.your_model_metrics['fn']:6d}")
        print(f"  True Positives:  {self.your_model_metrics['tp']:6d}")
        
        print()
        
        # =====================================================================
        # DATA DISTRIBUTION
        # =====================================================================
        
        print("DATA DISTRIBUTION")
        print("-" * 100)
        
        total = self.baseline_metrics['n_samples']
        n_pos = self.baseline_metrics['n_positive']
        n_neg = self.baseline_metrics['n_negative']
        
        print(f"Total samples:     {total}")
        print(f"Positive samples:  {n_pos} ({n_pos/total*100:.1f}%)")
        print(f"Negative samples:  {n_neg} ({n_neg/total*100:.1f}%)")
        
        print()
        
        # =====================================================================
        # HEAD-TO-HEAD COMPARISON
        # =====================================================================
        
        print("HEAD-TO-HEAD COMPARISON")
        print("-" * 100)
        
        comp = self.comparison
        
        print(f"Total predictions:       {total}")
        print()
        print(f"Models agree:            {comp['total_agreement']} ({comp['agreement_rate']*100:.1f}%)")
        print(f"  Both correct:          {comp['both_correct_when_agree']}")
        print(f"  Both wrong:            {comp['both_wrong_when_agree']}")
        print()
        print(f"Models disagree:         {comp['total_disagreement']}")
        print(f"  Baseline wins:         {comp['baseline_wins']}")
        print(f"  Your Model wins:       {comp['your_model_wins']}")
        print(f"  Net improvement:       {comp['net_improvement']:+d}")
        print()
        print(f"Baseline accuracy:       {comp['baseline_accuracy']:.4f} ({comp['baseline_accuracy']*100:.2f}%)")
        print(f"Your Model accuracy:     {comp['your_model_accuracy']:.4f} ({comp['your_model_accuracy']*100:.2f}%)")
        print(f"Improvement:             {comp['accuracy_improvement']:+.4f} ({comp['accuracy_improvement_pct']:+.2f}%)")
        
        print()
        
        # =====================================================================
        # SUMMARY
        # =====================================================================
        
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)
        
        if comp['your_model_wins'] > comp['baseline_wins']:
            print(f"✓ Your Model is BETTER")
            print(f"  - Outperforms baseline by {comp['your_model_wins']} predictions")
            print(f"  - Accuracy improvement: {comp['accuracy_improvement_pct']:.2f}%")
        elif comp['baseline_wins'] > comp['your_model_wins']:
            print(f"⚠ Baseline is BETTER")
            print(f"  - Your model needs improvement")
        else:
            print(f"= Models are EQUIVALENT")
        
        print()
    
    def get_metrics_dataframe(self):
        """Return metrics as pandas DataFrame for easy viewing"""
        
        if self.baseline_metrics is None or self.your_model_metrics is None:
            return None
        
        # Create list of metric rows
        rows = []
        
        metrics_to_include = [
            ('Accuracy', 'accuracy'),
            ('Precision', 'precision'),
            ('Recall', 'recall'),
            ('F1-Score', 'f1'),
            ('Balanced Accuracy', 'balanced_accuracy'),
            ('ROC-AUC', 'roc_auc'),
            ('Sensitivity', 'sensitivity'),
            ('Specificity', 'specificity'),
        ]
        
        for metric_name, metric_key in metrics_to_include:
            baseline_val = self.baseline_metrics.get(metric_key)
            your_model_val = self.your_model_metrics.get(metric_key)
            
            if baseline_val is None or your_model_val is None:
                continue
            
            rows.append({
                'Metric': metric_name,
                'Baseline': baseline_val,
                'Your Model': your_model_val,
                'Improvement': your_model_val - baseline_val
            })
        
        return pd.DataFrame(rows)
    
    def get_confusion_matrix_summary(self):
        """Return confusion matrix summary"""
        
        if self.baseline_metrics is None or self.your_model_metrics is None:
            return None
        
        return {
            'baseline': {
                'tn': self.baseline_metrics['tn'],
                'fp': self.baseline_metrics['fp'],
                'fn': self.baseline_metrics['fn'],
                'tp': self.baseline_metrics['tp'],
            },
            'your_model': {
                'tn': self.your_model_metrics['tn'],
                'fp': self.your_model_metrics['fp'],
                'fn': self.your_model_metrics['fn'],
                'tp': self.your_model_metrics['tp'],
            }
        }
    
    def export_metrics_to_csv(self, filepath):
        """Export metrics to CSV"""
        
        df = self.get_metrics_dataframe()
        if df is not None:
            df.to_csv(filepath, index=False)
            print(f"✓ Metrics exported to {filepath}")
        else:
            print("No metrics to export")
    
    def export_report(self, filepath):
        """Export full report to text file"""
        
        with open(filepath, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 100 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Metrics table
            f.write("CLASSIFICATION METRICS\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Metric':<25} {'Baseline':<20} {'Your Model':<20} {'Improvement':<20}\n")
            f.write("-" * 100 + "\n")
            
            metrics_to_show = [
                ('Accuracy', 'accuracy'),
                ('Precision', 'precision'),
                ('Recall', 'recall'),
                ('F1-Score', 'f1'),
                ('Balanced Accuracy', 'balanced_accuracy'),
            ]
            
            for metric_name, metric_key in metrics_to_show:
                baseline_val = self.baseline_metrics.get(metric_key)
                your_model_val = self.your_model_metrics.get(metric_key)
                
                if baseline_val is None or your_model_val is None:
                    continue
                
                improvement = your_model_val - baseline_val
                sign = "+" if improvement >= 0 else ""
                
                f.write(f"{metric_name:<25} {baseline_val:<20.4f} {your_model_val:<20.4f} {sign}{improvement:<19.4f}\n")
            
            f.write("\n")
            
            # Comparison
            f.write("HEAD-TO-HEAD COMPARISON\n")
            f.write("-" * 100 + "\n")
            
            comp = self.comparison
            f.write(f"Total predictions:       {self.baseline_metrics['n_samples']}\n")
            f.write(f"Baseline wins:           {comp['baseline_wins']}\n")
            f.write(f"Your Model wins:         {comp['your_model_wins']}\n")
            f.write(f"Net improvement:         {comp['net_improvement']:+d}\n")
            f.write(f"Accuracy improvement:    {comp['accuracy_improvement_pct']:+.2f}%\n")
        
        print(f"✓ Report exported to {filepath}")
