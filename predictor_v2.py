"""
Jellyfish Forecasting Predictor - Clean Design

Two models only:
  1. Baseline: Simple logistic regression (benchmark)
  2. Your Model: Main neural network (your design choice)

Provides clear comparison without clutter.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import timedelta
from data_loader_forecasting import load_jellyfish_data
from models import BaselineLogisticRegression, HybridNet  # Your best model


def create_engineered_features_forecasting(X, lookback=7):
    """Create engineered features for baseline logistic regression"""
    
    n_samples, lookback, n_features = X.shape
    engineered_features = []
    
    for sample_idx in range(n_samples):
        seq = X[sample_idx]
        features_for_sample = []
        
        for feat_idx in range(n_features):
            time_series = seq[:, feat_idx]
            
            current = time_series[-1]
            features_for_sample.append(current)
            
            prev = time_series[-2] if lookback >= 2 else time_series[0]
            features_for_sample.append(prev)
            
            prev_3 = time_series[-4] if lookback >= 4 else time_series[0]
            features_for_sample.append(prev_3)
            
            x_vals = np.arange(lookback)
            coeffs = np.polyfit(x_vals, time_series, 1)
            trend = coeffs[0]
            features_for_sample.append(trend)
            
            mean_val = np.mean(time_series)
            features_for_sample.append(mean_val)
            
            std_val = np.std(time_series)
            features_for_sample.append(std_val)
            
            min_val = np.min(time_series)
            max_val = np.max(time_series)
            features_for_sample.append(min_val)
            features_for_sample.append(max_val)
            
            change_1day = current - prev
            features_for_sample.append(change_1day)
            
            change_3day = current - prev_3
            features_for_sample.append(change_3day)
        
        engineered_features.append(features_for_sample)
    
    return np.array(engineered_features, dtype=np.float32)


class JellyfishPredictor:
    """
    Clean predictor interface comparing Baseline vs Your Model
    
    Two models only:
    1. Baseline: Logistic regression (simple benchmark)
    2. Your Model: Hybrid CNN+LSTM (your design)
    """
    
    def __init__(self, device='cpu'):
        """Initialize predictor"""
        self.device = torch.device(device)
        self.baseline_model = None
        self.your_model = None
        self.normalization_stats = {}
        self.data_cache = None
        
        print(f"Initialized JellyfishPredictor on {device}")
    
    def load_data_cache(self, lookback_days=7, forecast_days=1):
        """Load and cache all data"""
        print("Loading data cache...")
        X, y, metadata = load_jellyfish_data(lookback_days, forecast_days)
        
        # Compute normalization
        X_tensor = torch.FloatTensor(X)
        mean = X_tensor.mean(dim=0)
        std = X_tensor.std(dim=0)

        X_engineered = create_engineered_features_forecasting(X, lookback=lookback_days)
        X_eng_tensor = torch.FloatTensor(X_engineered)
        mean_eng = X_eng_tensor.mean(dim=0)
        std_eng = X_eng_tensor.std(dim=0)
        
        self.normalization_stats['mean'] = mean
        self.normalization_stats['std'] = std
        self.normalization_stats['mean_eng'] = mean_eng
        self.normalization_stats['std_eng'] = std_eng
        self.normalization_stats['lookback_days'] = lookback_days
        
        self.data_cache = {
            'X': X,
            'y': y,
            'metadata': metadata,
            'X_tensor': X_tensor
        }
        
        print(f"✓ Data cache loaded: {X.shape}\n")
    
    def load_baseline_model(self, model_path):
        """Load baseline logistic regression"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        input_dim = 110
        linear_weight = state_dict.get('linear.weight') if isinstance(state_dict, dict) else None
        if isinstance(linear_weight, torch.Tensor) and linear_weight.ndim == 2:
            input_dim = int(linear_weight.shape[1])

        self.baseline_model = BaselineLogisticRegression(input_dim=input_dim)
        self.baseline_model.load_state_dict(state_dict)
        self.baseline_model.to(self.device)
        self.baseline_model.eval()
        print(f"✓ Loaded Baseline model from {model_path}")
    
    def load_your_model(self, model_path):
        """Load your main model (Hybrid CNN+LSTM)"""
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.your_model = HybridNet(input_dim=11)
        self.your_model.load_state_dict(state_dict)
        self.your_model.to(self.device)
        self.your_model.eval()
        print(f"✓ Loaded Your Model from {model_path}")
    
    def predict_sequence(self, X_sequence, model_type):
        """Predict for a single sequence"""
        
        if model_type == 'baseline':
            if self.baseline_model is None:
                raise RuntimeError("Baseline model not loaded")
            model = self.baseline_model
        elif model_type == 'your_model':
            if self.your_model is None:
                raise RuntimeError("Your model not loaded")
            model = self.your_model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Prepare input
        if isinstance(X_sequence, np.ndarray):
            X_tensor = torch.FloatTensor(X_sequence)
        else:
            X_tensor = X_sequence
        
        # Add batch dimension if needed
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)
        elif X_tensor.dim() == 2 and X_tensor.shape[0] != 1:
            X_tensor = X_tensor.unsqueeze(0)
        
        X_tensor = X_tensor.to(self.device)
        
        with torch.no_grad():
            output = model(X_tensor)
            probability = output.cpu().item()
        
        return probability
    
    def predict_for_beach_date(self, beach_id, forecast_date, show_details=False):
        """
        Make prediction for a specific beach and date
        
        Returns comparison of Baseline vs Your Model
        
        Args:
            beach_id: Beach ID
            forecast_date: Date to forecast (str or date)
            show_details: Print detailed comparison
        
        Returns:
            dict with both model predictions
        """
        if self.data_cache is None:
            raise RuntimeError("Call load_data_cache() first")
        
        # Parse date if string
        if isinstance(forecast_date, str):
            from datetime import datetime
            forecast_date = datetime.strptime(forecast_date, '%Y-%m-%d').date()
        
        metadata = self.data_cache['metadata']
        
        # Find matching sequence
        match = metadata[
            (metadata['beach_id'] == beach_id) &
            (metadata['forecast_date'] == forecast_date)
        ]
        
        if len(match) == 0:
            return {
                'beach_id': beach_id,
                'beach_name': 'Unknown',
                'forecast_date': forecast_date,
                'error': f'No data for beach {beach_id} on {forecast_date}'
            }
        
        match_idx = match.index[0]
        X_sequence = self.data_cache['X'][match_idx]
        beach_name = match.iloc[0]['beach_name']
        actual = match.iloc[0]['jellyfish_observed']
        
        # Get baseline prediction (engineered features)
        X_eng = create_engineered_features_forecasting(
            self.data_cache['X'][match_idx:match_idx+1], 
            lookback=self.normalization_stats['lookback_days']
        )
        X_eng_tensor = torch.FloatTensor(X_eng[0])
        mean_eng = self.normalization_stats['mean_eng']
        std_eng = self.normalization_stats['std_eng']
        X_eng_normalized = (X_eng_tensor - mean_eng) / (std_eng + 1e-8)
        baseline_prob = self.predict_sequence(X_eng_normalized, 'baseline')
        
        # Get your model prediction (normalized sequence)
        X_norm = torch.FloatTensor(X_sequence)
        mean = self.normalization_stats['mean']
        std = self.normalization_stats['std']
        X_normalized = (X_norm - mean) / (std + 1e-8)
        your_model_prob = self.predict_sequence(X_normalized, 'your_model')
        
        # Determine predictions
        baseline_pred = 'Yes' if baseline_prob > 0.5 else 'No'
        your_model_pred = 'Yes' if your_model_prob > 0.5 else 'No'
        actual_str = 'Yes' if actual == 1 else 'No'
        
        # Calculate confidence
        def get_confidence(prob):
            distance = abs(prob - 0.5)
            if distance > 0.3:
                return 'High'
            elif distance > 0.1:
                return 'Medium'
            else:
                return 'Low'
        
        baseline_conf = get_confidence(baseline_prob)
        your_model_conf = get_confidence(your_model_prob)
        
        result = {
            'beach_id': beach_id,
            'beach_name': beach_name,
            'forecast_date': forecast_date,
            'actual': actual_str,
            
            'baseline': {
                'probability': baseline_prob,
                'percentage': baseline_prob * 100,
                'prediction': baseline_pred,
                'confidence': baseline_conf
            },
            
            'your_model': {
                'probability': your_model_prob,
                'percentage': your_model_prob * 100,
                'prediction': your_model_pred,
                'confidence': your_model_conf
            }
        }
        
        if show_details:
            self._print_comparison(result)
        
        return result
    
    def _print_comparison(self, result):
        """Print nicely formatted comparison"""
        print(f"\n" + "=" * 80)
        print(f"JELLYFISH PREDICTION - BASELINE vs YOUR MODEL")
        print(f"=" * 80)
        print(f"Beach:          {result['beach_id']} - {result['beach_name']}")
        print(f"Date:           {result['forecast_date']}")
        print(f"Actual Outcome: {result['actual']}")
        print(f"=" * 80)
        print()
        
        print(f"{'Model':<20} {'Probability':<15} {'Percentage':<15} {'Prediction':<15} {'Confidence':<15}")
        print("-" * 80)
        
        # Baseline
        b = result['baseline']
        print(f"{'Baseline':<20} {b['probability']:<15.4f} {b['percentage']:<15.2f}% {b['prediction']:<15} {b['confidence']:<15}")
        
        # Your Model
        y = result['your_model']
        print(f"{'Your Model':<20} {y['probability']:<15.4f} {y['percentage']:<15.2f}% {y['prediction']:<15} {y['confidence']:<15}")
        
        # Improvement
        print("-" * 80)
        improvement = y['probability'] - b['probability']
        improvement_pct = improvement * 100
        sign = "+" if improvement >= 0 else ""
        print(f"{'Improvement':<20} {improvement:<15.4f} {improvement_pct:<15.2f}%")
        
        print(f"=" * 80)
        print()
    
    def compare_multiple_predictions(self, predictions_list):
        """
        Compare baseline vs your model across multiple predictions
        
        Args:
            predictions_list: List of (beach_id, forecast_date) tuples
        
        Returns:
            DataFrame with comparison
        """
        results = []
        
        for beach_id, forecast_date in predictions_list:
            result = self.predict_for_beach_date(beach_id, forecast_date)
            
            if 'error' not in result:
                results.append({
                    'Beach ID': beach_id,
                    'Beach Name': result['beach_name'][:15],
                    'Date': str(result['forecast_date']),
                    'Actual': result['actual'],
                    'Baseline %': f"{result['baseline']['percentage']:.1f}%",
                    'Your Model %': f"{result['your_model']['percentage']:.1f}%",
                    'Your Model Better': "✓" if result['your_model']['probability'] > result['baseline']['probability'] else ""
                })
        
        return pd.DataFrame(results)
    
    def get_model_agreement(self, beach_id, forecast_date):
        """Check if baseline and your model agree"""
        result = self.predict_for_beach_date(beach_id, forecast_date)
        
        if 'error' in result:
            return None
        
        baseline_pred = result['baseline']['prediction']
        your_model_pred = result['your_model']['prediction']
        actual = result['actual']
        
        agreement = {
            'baseline': baseline_pred,
            'your_model': your_model_pred,
            'actual': actual,
            'agree': baseline_pred == your_model_pred,
            'both_correct': (baseline_pred == actual) and (your_model_pred == actual),
            'only_yours_correct': (your_model_pred == actual) and (baseline_pred != actual),
            'only_baseline_correct': (baseline_pred == actual) and (your_model_pred != actual),
            'both_wrong': (baseline_pred != actual) and (your_model_pred != actual)
        }
        
        return agreement


# Example usage
if __name__ == '__main__':
    # Initialize
    predictor = JellyfishPredictor(device='cpu')
    
    # Load data
    predictor.load_data_cache(lookback_days=7, forecast_days=1)
    
    # Show available dates
    metadata = predictor.data_cache['metadata']
    print("Sample beach-date combinations available:")
    print(metadata[['beach_id', 'beach_name', 'forecast_date', 'jellyfish_observed']].head(10))
    print()
    
    print("=" * 80)
    print("TO USE THIS PREDICTOR:")
    print("=" * 80)
    print("""
1. Load Models:
   predictor.load_baseline_model('baseline_model.pth')
   predictor.load_your_model('hybrid_model.pth')

2. Make Single Prediction:
   result = predictor.predict_for_beach_date(beach_id=1, 
                                            forecast_date='2011-09-19',
                                            show_details=True)
   
   Returns comparison of Baseline vs Your Model

3. Compare Multiple Predictions:
   predictions = [(1, '2011-09-19'), (2, '2011-10-23'), (3, '2011-08-13')]
   df = predictor.compare_multiple_predictions(predictions)
   print(df)

4. Check Agreement:
   agreement = predictor.get_model_agreement(beach_id=1, 
                                            forecast_date='2011-09-19')
   print(f"Models agree: {agreement['agree']}")
   print(f"Both correct: {agreement['both_correct']}")
   print(f"Your model better: {agreement['only_yours_correct']}")
    """)
