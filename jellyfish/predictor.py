"""
Jellyfish Forecasting Predictor

Provides a clean interface to make predictions:
  - Given a beach_id and date
  - Returns probability of jellyfish presence (0-100%)
  - Can load trained models and make real-time predictions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import timedelta, date
from .data_loader import load_jellyfish_data
from .models import (
    BaselineLogisticRegression,
    FeedforwardNet,
    LSTMNet,
    GRUNet,
    Conv1DNet,
    HybridNet
)


def create_engineered_features_forecasting(X, lookback=7):
    """Create engineered features for baseline logistic regression
    
    Takes historical sequences and extracts temporal features:
    - Latest values (most recent day)
    - Temporal trends (linear slope over lookback period)
    - Volatility (standard deviation over period)
    - Lagged differences (day-to-day changes)
    
    Args:
        X: Shape (n_samples, lookback, n_features)
        lookback: Number of days in lookback window (typically 7)
    
    Returns:
        X_engineered: Shape (n_samples, engineered_dim)
    """
    
    n_samples, lookback, n_features = X.shape
    engineered_features = []
    
    for sample_idx in range(n_samples):
        seq = X[sample_idx]  # (lookback, n_features)
        features_for_sample = []
        
        for feat_idx in range(n_features):
            time_series = seq[:, feat_idx]  # (lookback,)
            
            # 1. Current value (most recent day)
            current = time_series[-1]
            features_for_sample.append(current)
            
            # 2. Previous day (t-1)
            prev = time_series[-2] if lookback >= 2 else time_series[0]
            features_for_sample.append(prev)
            
            # 3. 3 days ago (t-3)
            prev_3 = time_series[-4] if lookback >= 4 else time_series[0]
            features_for_sample.append(prev_3)
            
            # 4. Trend (linear slope)
            x_vals = np.arange(lookback)
            coeffs = np.polyfit(x_vals, time_series, 1)
            trend = coeffs[0]
            features_for_sample.append(trend)
            
            # 5. Mean over lookback period
            mean_val = np.mean(time_series)
            features_for_sample.append(mean_val)
            
            # 6. Std over lookback period (volatility)
            std_val = np.std(time_series)
            features_for_sample.append(std_val)
            
            # 7. Min/Max over period
            min_val = np.min(time_series)
            max_val = np.max(time_series)
            features_for_sample.append(min_val)
            features_for_sample.append(max_val)
            
            # 8. 1-day change (current - prev)
            change_1day = current - prev
            features_for_sample.append(change_1day)
            
            # 9. 3-day change (current - 3-day-ago)
            change_3day = current - prev_3
            features_for_sample.append(change_3day)
        
        engineered_features.append(features_for_sample)
    
    return np.array(engineered_features, dtype=np.float32)


class JellyfishPredictor:
    """Unified predictor interface for jellyfish forecasting
    
    Provides methods to:
    1. Load trained models
    2. Make predictions for specific beach-date combinations
    3. Get probabilities and confidence intervals
    """
    
    def __init__(self, device='cpu'):
        """Initialize predictor
        
        Args:
            device: torch device ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.models = {}
        self.normalization_stats = {}
        self.data_cache = None
        
        print(f"Initialized JellyfishPredictor on {device}")
    
    def load_data_cache(self, lookback_days=7, forecast_days=1):
        """Load and cache all data for faster predictions
        
        Args:
            lookback_days: Historical window (default: 7)
            forecast_days: Forecast horizon (default: 1)
        """
        print("Loading data cache...")
        X, y, metadata = load_jellyfish_data(lookback_days, forecast_days)
        
        # Compute normalization statistics
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
        
        print(f"✓ Data cache loaded: {X.shape}")
    
    def load_model(self, model_name, model_path, input_dim=None):
        """Load a trained model
        
        Args:
            model_name: Name of model ('Baseline', 'LSTM', 'GRU', 'Conv1D', 'Hybrid', 'Feedforward')
            model_path: Path to saved model weights
            input_dim: Input dimension (only for Baseline)
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        if model_name == 'Baseline':
            inferred_input_dim = None
            linear_weight = state_dict.get('linear.weight') if isinstance(state_dict, dict) else None
            if isinstance(linear_weight, torch.Tensor) and linear_weight.ndim == 2:
                inferred_input_dim = int(linear_weight.shape[1])

            if input_dim is None:
                input_dim = inferred_input_dim if inferred_input_dim is not None else 110
            elif inferred_input_dim is not None and input_dim != inferred_input_dim:
                print(
                    f"⚠ Baseline input_dim={input_dim} does not match checkpoint "
                    f"({inferred_input_dim}). Using checkpoint value."
                )
                input_dim = inferred_input_dim

            model = BaselineLogisticRegression(input_dim)
        elif model_name == 'Feedforward':
            inferred_input_dim = None
            fc1_weight = state_dict.get('fc1.weight') if isinstance(state_dict, dict) else None
            if isinstance(fc1_weight, torch.Tensor) and fc1_weight.ndim == 2:
                inferred_input_dim = int(fc1_weight.shape[1])

            if input_dim is None:
                input_dim = inferred_input_dim if inferred_input_dim is not None else 7 * 11
            elif inferred_input_dim is not None and input_dim != inferred_input_dim:
                print(
                    f"⚠ Feedforward input_dim={input_dim} does not match checkpoint "
                    f"({inferred_input_dim}). Using checkpoint value."
                )
                input_dim = inferred_input_dim
            model = FeedforwardNet(input_dim)
        elif model_name == 'LSTM':
            model = LSTMNet(input_dim=11)
        elif model_name == 'GRU':
            model = GRUNet(input_dim=11)
        elif model_name == 'Conv1D':
            model = Conv1DNet(input_dim=11)
        elif model_name == 'Hybrid':
            model = HybridNet(input_dim=11)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Load weights
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        self.models[model_name] = model
        print(f"✓ Loaded {model_name} model from {model_path}")
    
    def predict_sequence(self, X_sequence, model_name, use_baseline=False):
        """Predict for a single sequence
        
        Args:
            X_sequence: Input sequence (7, 11) or engineered features (99,)
            model_name: Which model to use
            use_baseline: Whether to use baseline (requires engineered features)
        
        Returns:
            probability (float): Probability of jellyfish presence (0-1)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Call load_model() first.")
        
        model = self.models[model_name]
        
        # Prepare input
        if isinstance(X_sequence, np.ndarray):
            X_tensor = torch.FloatTensor(X_sequence)
        else:
            X_tensor = X_sequence
        
        # Add batch dimension if needed
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)  # (engineered_dim,) → (1, engineered_dim)
        elif X_tensor.dim() == 2 and X_tensor.shape[0] != 1:
            # Assume it's (7, 11) without batch dimension
            X_tensor = X_tensor.unsqueeze(0)  # (7, 11) → (1, 7, 11)
        
        X_tensor = X_tensor.to(self.device)
        
        with torch.no_grad():
            output = model(X_tensor)
            probability = output.cpu().item()
        
        return probability

    @staticmethod
    def _feature_indices():
        return {
            'latitude': 0,
            'longitude': 1,
            'month': 2,
            'day_of_month': 3,
            'sin_month': 4,
            'cos_month': 5,
            'distance_from_coast': 6,
            'species_id': 7,
            'diameter_cm': 8,
            'observation_count': 9,
            'sting': 10,
        }

    @staticmethod
    def _project_next_from_recent(values, clamp_min=None, clamp_max=None):
        """Project next value using a linear trend over recent values."""
        if values.size == 0:
            return 0.0
        if values.size == 1:
            next_val = float(values[-1])
        else:
            x = np.arange(values.size, dtype=np.float32)
            slope, intercept = np.polyfit(x, values.astype(np.float32), 1)
            next_val = float(slope * values.size + intercept)

        if clamp_min is not None:
            next_val = max(clamp_min, next_val)
        if clamp_max is not None:
            next_val = min(clamp_max, next_val)
        return next_val

    def _build_future_sequence(self, beach_id, forecast_date):
        """Construct a forecast sequence for a future date using latest known beach history.

        Strategy:
        - Use latest available sequence for this beach from cached metadata
        - Roll sequence forward day-by-day until forecast_date
        - Update dynamic non-calendar features using recent trend
        - Keep static geographic features fixed
        - Recompute calendar features (month/day/sin/cos) each rolled day
        """
        metadata = self.data_cache['metadata']
        beach_rows = metadata[metadata['beach_id'] == beach_id].copy()

        if beach_rows.empty:
            return None, None, None

        beach_rows = beach_rows.sort_values('forecast_date').reset_index()
        latest_row = beach_rows.iloc[-1]
        latest_idx = int(latest_row['index'])
        latest_date = latest_row['forecast_date']
        beach_name = latest_row['beach_name']

        if forecast_date <= latest_date:
            return None, beach_name, latest_date

        seq = self.data_cache['X'][latest_idx].copy().astype(np.float32)
        days_ahead = (forecast_date - latest_date).days

        idx = self._feature_indices()
        for step in range(1, days_ahead + 1):
            target_day = latest_date + timedelta(days=step)
            new_row = seq[-1].copy()

            # Keep geographic/static beach features constant
            new_row[idx['latitude']] = seq[-1, idx['latitude']]
            new_row[idx['longitude']] = seq[-1, idx['longitude']]
            new_row[idx['distance_from_coast']] = seq[-1, idx['distance_from_coast']]

            # Species index behaves like a categorical proxy; keep last known value
            new_row[idx['species_id']] = seq[-1, idx['species_id']]

            # Dynamic features: project from recent trend
            recent_diameter = seq[:, idx['diameter_cm']]
            recent_obs_count = seq[:, idx['observation_count']]
            recent_sting = seq[:, idx['sting']]

            new_row[idx['diameter_cm']] = self._project_next_from_recent(recent_diameter, clamp_min=0.0)
            new_row[idx['observation_count']] = self._project_next_from_recent(recent_obs_count, clamp_min=0.0)
            new_row[idx['sting']] = self._project_next_from_recent(recent_sting, clamp_min=0.0, clamp_max=1.0)

            month = float(target_day.month)
            day_of_month = float(target_day.day)
            sin_month = float(np.sin(2 * np.pi * month / 12.0))
            cos_month = float(np.cos(2 * np.pi * month / 12.0))

            new_row[idx['month']] = month
            new_row[idx['day_of_month']] = day_of_month
            new_row[idx['sin_month']] = sin_month
            new_row[idx['cos_month']] = cos_month

            seq = np.vstack([seq[1:], new_row]).astype(np.float32)

        return seq, beach_name, latest_date
    
    def predict_for_beach_date(self, beach_id, forecast_date, model_name):
        """Make prediction for a specific beach and date
        
        Args:
            beach_id: Beach ID (integer)
            forecast_date: Date to forecast (datetime.date or str 'YYYY-MM-DD')
            model_name: Which model to use
        
        Returns:
            dict: {
                'beach_id': int,
                'beach_name': str,
                'forecast_date': date,
                'probability': float (0-1),
                'percentage': float (0-100),
                'prediction': str ('Yes' or 'No'),
                'confidence': str ('High', 'Medium', 'Low')
            }
        """
        if self.data_cache is None:
            raise RuntimeError("Call load_data_cache() first")
        
        # Parse date if string
        if isinstance(forecast_date, str):
            from datetime import datetime
            forecast_date = datetime.strptime(forecast_date, '%Y-%m-%d').date()
        
        metadata = self.data_cache['metadata']
        
        # Find matching sequence in metadata
        match = metadata[
            (metadata['beach_id'] == beach_id) &
            (metadata['forecast_date'] == forecast_date)
        ]

        used_extrapolation = False
        extrapolated_from_date = None

        if len(match) == 0:
            X_sequence, beach_name, latest_date = self._build_future_sequence(beach_id, forecast_date)
            if X_sequence is None:
                return {
                    'beach_id': beach_id,
                    'beach_name': beach_name if beach_name is not None else 'Unknown',
                    'forecast_date': forecast_date,
                    'probability': None,
                    'percentage': None,
                    'prediction': 'No data',
                    'confidence': 'N/A',
                    'error': f'No data found for beach {beach_id} on {forecast_date}'
                }
            used_extrapolation = True
            extrapolated_from_date = latest_date
        else:
            # Get the corresponding sequence
            match_idx = match.index[0]
            X_sequence = self.data_cache['X'][match_idx]  # (lookback, n_features)
            beach_name = match.iloc[0]['beach_name']
        
        # Normalize if using neural network
        if model_name != 'Baseline':
            X_tensor = torch.FloatTensor(X_sequence)
            mean = self.normalization_stats['mean']
            std = self.normalization_stats['std']
            X_normalized = (X_tensor - mean) / (std + 1e-8)
            probability = self.predict_sequence(X_normalized, model_name)
        else:
            # Use baseline with engineered features
            X_eng = create_engineered_features_forecasting(
                X_sequence[np.newaxis, ...],
                lookback=self.normalization_stats['lookback_days']
            )
            X_eng_tensor = torch.FloatTensor(X_eng[0])
            mean_eng = self.normalization_stats['mean_eng']
            std_eng = self.normalization_stats['std_eng']
            X_eng_normalized = (X_eng_tensor - mean_eng) / (std_eng + 1e-8)
            probability = self.predict_sequence(X_eng_normalized, model_name)
        
        # Determine prediction and confidence
        percentage = probability * 100
        prediction = 'Yes' if probability > 0.5 else 'No'
        
        # Confidence based on how far from 0.5
        distance_from_threshold = abs(probability - 0.5)
        if distance_from_threshold > 0.3:
            confidence = 'High'
        elif distance_from_threshold > 0.1:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        return {
            'beach_id': beach_id,
            'beach_name': beach_name,
            'forecast_date': forecast_date,
            'probability': probability,
            'percentage': percentage,
            'prediction': prediction,
            'confidence': confidence,
            'actual': match.iloc[0]['jellyfish_observed'] if (len(match) > 0 and 'jellyfish_observed' in match.columns) else None,
            'extrapolated': used_extrapolation,
            'extrapolated_from_date': extrapolated_from_date,
            'extrapolation_method': 'seasonal + trend projection' if used_extrapolation else 'historical match',
        }

    def predict_days_ahead(self, beach_id, days_ahead, model_name, start_date=None):
        """Predict for beach_id at N days ahead from start_date (or today).

        Args:
            beach_id: Beach ID (integer)
            days_ahead: Non-negative integer number of days in the future
            model_name: Loaded model name
            start_date: Optional date (datetime.date or 'YYYY-MM-DD'), defaults to today

        Returns:
            dict: Same schema as predict_for_beach_date
        """
        if days_ahead < 0:
            raise ValueError("days_ahead must be non-negative")

        if start_date is None:
            base_date = date.today()
        elif isinstance(start_date, str):
            from datetime import datetime
            base_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        else:
            base_date = start_date

        forecast_date = base_date + timedelta(days=int(days_ahead))
        result = self.predict_for_beach_date(beach_id, forecast_date, model_name)
        result['days_ahead'] = int(days_ahead)
        result['base_date'] = base_date
        return result
    
    def predict_multiple(self, predictions_list, model_name):
        """Make multiple predictions
        
        Args:
            predictions_list: List of (beach_id, forecast_date) tuples
            model_name: Which model to use
        
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        for beach_id, forecast_date in predictions_list:
            result = self.predict_for_beach_date(beach_id, forecast_date, model_name)
            results.append(result)
        
        return results
    
    def predict_all_models(self, beach_id, forecast_date):
        """Make predictions using all loaded models
        
        Args:
            beach_id: Beach ID
            forecast_date: Date to forecast
        
        Returns:
            dict: Predictions from all models
        """
        results = {}
        for model_name in self.models.keys():
            result = self.predict_for_beach_date(beach_id, forecast_date, model_name)
            results[model_name] = result
        
        return results
    
    def compare_predictions(self, beach_id, forecast_date):
        """Display side-by-side comparison of all model predictions
        
        Args:
            beach_id: Beach ID
            forecast_date: Date to forecast
        """
        results = self.predict_all_models(beach_id, forecast_date)
        
        # Check if we have valid predictions
        valid_results = {k: v for k, v in results.items() if v.get('error') is None}
        
        if not valid_results:
            print(f"\n❌ No predictions available for beach {beach_id} on {forecast_date}")
            if results:
                print(f"Error: {list(results.values())[0]['error']}")
            return
        
        single_model_mode = len(valid_results) == 1
        first_result = list(valid_results.values())[0]
        print(f"\n" + "=" * 80)
        if single_model_mode:
            print(f"JELLYFISH FORECAST RESULT")
        else:
            print(f"JELLYFISH FORECAST COMPARISON")
        print(f"=" * 80)
        print(f"Beach ID:        {beach_id}")
        print(f"Beach Name:      {first_result['beach_name']}")
        print(f"Forecast Date:   {forecast_date}")
        if first_result.get('extrapolated') or first_result.get('actual') is None:
            actual_outcome = 'Unknown (future forecast)'
        else:
            actual_outcome = 'Jellyfish Present' if first_result.get('actual') == 1 else 'No Jellyfish'
        print(f"Actual Outcome:  {actual_outcome}")
        print(f"=" * 80)
        print()
        
        print(f"{'Model':<20} {'Probability':<15} {'Percentage':<15} {'Prediction':<15} {'Confidence':<15}")
        print("-" * 80)
        
        for model_name in sorted(valid_results.keys()):
            result = valid_results[model_name]
            prob = result['probability']
            pct = result['percentage']
            pred = result['prediction']
            conf = result['confidence']
            pct_display = f"{pct:.2f}%"
            
            print(f"{model_name:<20} {prob:<15.4f} {pct_display:<15} {pred:<15} {conf:<15}")

        if not single_model_mode:
            # Calculate ensemble prediction (average)
            avg_prob = np.mean([v['probability'] for v in valid_results.values()])
            avg_percentage = avg_prob * 100
            ensemble_pred = 'Yes' if avg_prob > 0.5 else 'No'
            avg_percentage_display = f"{avg_percentage:.2f}%"

            print("-" * 80)
            print(f"{'Ensemble':<20} {avg_prob:<15.4f} {avg_percentage_display:<15} {ensemble_pred:<15}")
        print("=" * 80)
        print()


# Example usage
if __name__ == '__main__':
    # Initialize predictor
    predictor = JellyfishPredictor(device='cpu')
    
    # Load data
    predictor.load_data_cache(lookback_days=7, forecast_days=1)
    
    # Example: Get some metadata to find valid beach-date combinations
    metadata = predictor.data_cache['metadata']
    print("\nSample beach-date combinations from dataset:")
    print(metadata[['beach_id', 'beach_name', 'forecast_date', 'jellyfish_observed']].head(10))
    
    # Example prediction (you would replace with actual trained model paths)
    print("\n" + "=" * 80)
    print("To use the predictor:")
    print("=" * 80)
    print("""
1. Load trained models:
   predictor.load_model('LSTM', 'path/to/lstm_model.pth')
    predictor.load_model('Baseline', 'path/to/baseline_model.pth')

2. Make a single prediction:
   result = predictor.predict_for_beach_date(beach_id=5, 
                                             forecast_date='2025-03-15',
                                             model_name='LSTM')
   print(f"Probability: {result['percentage']:.2f}%")
   print(f"Prediction: {result['prediction']}")

3. Compare all models:
   predictor.compare_predictions(beach_id=5, forecast_date='2025-03-15')

4. Make multiple predictions:
   predictions = [
       (5, '2025-03-15'),
       (7, '2025-03-16'),
       (10, '2025-03-17')
   ]
   results = predictor.predict_multiple(predictions, model_name='LSTM')
    """)
