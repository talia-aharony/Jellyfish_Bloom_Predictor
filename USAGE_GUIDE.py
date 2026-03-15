"""
JELLYFISH FORECASTING SYSTEM - COMPLETE USAGE GUIDE

Clean, elegant architecture for predicting jellyfish presence by beach and date.
"""

# =============================================================================
# FILE STRUCTURE
# =============================================================================

"""
├── models.py                          # All neural network models
│   ├── BaselineLogisticRegression    # Simple logistic regression baseline
│   ├── FeedforwardNet                 # Dense feedforward network
│   ├── LSTMNet                        # LSTM sequence model
│   ├── GRUNet                         # GRU sequence model
│   ├── Conv1DNet                      # 1D CNN model
│   └── HybridNet                      # CNN + LSTM hybrid
│
├── data_loader_forecasting.py         # Data loading and aggregation
│   └── load_jellyfish_data()          # Load per-beach per-day data
│
├── train.py                           # Training script
│   ├── Trainer                        # Training pipeline
│   ├── plot_training_history()        # Visualization
│   └── train_all_models()             # Main training function
│
├── predictor.py                       # Prediction interface
│   ├── JellyfishPredictor             # Main prediction class
│   └── create_engineered_features_forecasting()  # Feature engineering
│
└── predict_example.py                 
"""

# =============================================================================
# STEP 1: TRAIN MODELS (One-time setup)
# =============================================================================

"""
python train.py

This will:
1. Load data with per-beach per-day aggregation
2. Train baseline logistic regression
3. Train 5 neural network models (Feedforward, LSTM, GRU, Conv1D, Hybrid)
4. Save trained weights:
   - baseline_model.pth
   - feedforward_model.pth
   - lstm_model.pth
   - gru_model.pth
   - conv1d_model.pth
   - hybrid_model.pth
5. Plot training curves for each model
"""

# =============================================================================
# STEP 2: USE PREDICTOR (Make predictions)
# =============================================================================

# Example: predict_example.py

from predictor import JellyfishPredictor
from datetime import date

# Initialize predictor
predictor = JellyfishPredictor(device='cpu')

# Load data cache
predictor.load_data_cache(lookback_days=7, forecast_days=1)

# Load trained models
predictor.load_model('Baseline', 'baseline_model.pth', input_dim=99)
predictor.load_model('LSTM', 'lstm_model.pth')
predictor.load_model('Hybrid', 'hybrid_model.pth')

# =============================================================================
# METHOD 1: Single Prediction
# =============================================================================

result = predictor.predict_for_beach_date(
    beach_id=5,
    forecast_date='2025-03-15',
    model_name='LSTM'
)

print(f"Beach: {result['beach_name']}")
print(f"Date: {result['forecast_date']}")
print(f"Probability of Jellyfish: {result['percentage']:.1f}%")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")

"""
Output:
Beach: Tel Aviv Beach
Date: 2025-03-15
Probability of Jellyfish: 72.3%
Prediction: Yes
Confidence: High
"""

# =============================================================================
# METHOD 2: Compare All Models
# =============================================================================

predictor.compare_predictions(beach_id=5, forecast_date='2025-03-15')

"""
Output:
================================================================================
JELLYFISH FORECAST COMPARISON
================================================================================
Beach ID:        5
Beach Name:      Tel Aviv Beach
Forecast Date:   2025-03-15
Actual Outcome:  Jellyfish Present
================================================================================

Model                Probability     Percentage         Prediction         Confidence        
--------------------------------------------------------------------------------
Baseline             0.4234          42.34%              No                  Low            
Conv1D               0.5123          51.23%              Yes                 Low            
Feedforward          0.6789          67.89%              Yes                 Medium         
GRU                  0.7234          72.34%              Yes                 High           
Hybrid               0.7856          78.56%              Yes                 High           
LSTM                 0.7523          75.23%              Yes                 High           
--------------------------------------------------------------------------------
Ensemble             0.7043          70.43%              Yes                
================================================================================
"""

# =============================================================================
# METHOD 3: Multiple Predictions
# =============================================================================

predictions_list = [
    (5, '2025-03-15'),   # Beach 5, March 15
    (7, '2025-03-16'),   # Beach 7, March 16
    (10, '2025-03-17'),  # Beach 10, March 17
]

results = predictor.predict_multiple(predictions_list, model_name='LSTM')

for result in results:
    if result.get('error'):
        print(f"Beach {result['beach_id']} {result['forecast_date']}: {result['error']}")
    else:
        print(f"Beach {result['beach_id']} ({result['beach_name']}) {result['forecast_date']}: "
              f"{result['percentage']:.1f}% - {result['prediction']}")

"""
Output:
Beach 5 (Tel Aviv) 2025-03-15: 75.2% - Yes
Beach 7 (Netanya) 2025-03-16: 42.1% - No
Beach 10 (Ashkelon) 2025-03-17: 68.9% - Yes
"""

# =============================================================================
# METHOD 4: Ensemble Predictions
# =============================================================================

# Get predictions from all models
all_results = predictor.predict_all_models(beach_id=5, forecast_date='2025-03-15')

# Create ensemble prediction
probabilities = [v['probability'] for v in all_results.values()]
ensemble_prob = np.mean(probabilities)
ensemble_pct = ensemble_prob * 100
ensemble_pred = 'Yes' if ensemble_prob > 0.5 else 'No'

print(f"\nEnsemble Prediction:")
print(f"  Probability: {ensemble_pct:.2f}%")
print(f"  Prediction: {ensemble_pred}")
print(f"  Based on: {len(all_results)} models")

# =============================================================================
# CLASS REFERENCE: JellyfishPredictor
# =============================================================================

"""
JellyfishPredictor(device='cpu')
    Initialize predictor on CPU or GPU
    
    Methods:
    --------
    
    load_data_cache(lookback_days=7, forecast_days=1)
        Load and cache all data for faster predictions
        Required before making predictions
    
    load_model(model_name, model_path, input_dim=None)
        Load a trained model
        
        Args:
            model_name: 'Baseline', 'Feedforward', 'LSTM', 'GRU', 'Conv1D', 'Hybrid'
            model_path: Path to saved .pth file
            input_dim: For Baseline only (default: 99)
        
        Example:
            predictor.load_model('LSTM', 'lstm_model.pth')
    
    predict_for_beach_date(beach_id, forecast_date, model_name)
        Predict for specific beach and date
        
        Args:
            beach_id: Integer beach ID
            forecast_date: datetime.date or 'YYYY-MM-DD' string
            model_name: Which model to use
        
        Returns:
            dict with keys:
                - beach_id: int
                - beach_name: str
                - forecast_date: date
                - probability: float (0-1)
                - percentage: float (0-100)
                - prediction: 'Yes' or 'No'
                - confidence: 'High', 'Medium', or 'Low'
                - actual: observed outcome (if available)
        
        Example:
            result = predictor.predict_for_beach_date(
                beach_id=5,
                forecast_date='2025-03-15',
                model_name='LSTM'
            )
            print(f"{result['percentage']:.1f}%")
    
    predict_multiple(predictions_list, model_name)
        Make multiple predictions
        
        Args:
            predictions_list: List of (beach_id, date) tuples
            model_name: Which model to use
        
        Returns:
            List of prediction dicts
        
        Example:
            results = predictor.predict_multiple([
                (5, '2025-03-15'),
                (7, '2025-03-16')
            ], model_name='LSTM')
    
    predict_all_models(beach_id, forecast_date)
        Get predictions from all loaded models
        
        Returns:
            dict with model_name -> prediction dict
    
    compare_predictions(beach_id, forecast_date)
        Display side-by-side comparison of all models (prints to console)
"""

# =============================================================================
# COMPLETE EXAMPLE SCRIPT: predict_example.py
# =============================================================================

"""
#!/usr/bin/env python3

from predictor import JellyfishPredictor
import pandas as pd

def main():
    # Initialize
    predictor = JellyfishPredictor(device='cpu')
    
    # Load data
    print("Loading data...")
    predictor.load_data_cache(lookback_days=7, forecast_days=1)
    
    # Load models
    print("Loading models...")
    models_to_load = [
        ('Baseline', 'baseline_model.pth', 99),
        ('LSTM', 'lstm_model.pth', None),
        ('Hybrid', 'hybrid_model.pth', None),
    ]
    
    for model_name, model_path, input_dim in models_to_load:
        try:
            if input_dim:
                predictor.load_model(model_name, model_path, input_dim=input_dim)
            else:
                predictor.load_model(model_name, model_path)
        except FileNotFoundError:
            print(f"Warning: Could not load {model_name} - {model_path} not found")
    
    # Make predictions
    print("\nMaking predictions...")
    
    # Example beaches and dates
    predictions = [
        (5, '2025-03-15'),   # Tel Aviv
        (7, '2025-03-16'),   # Netanya
        (10, '2025-03-17'),  # Ashkelon
    ]
    
    results = []
    for beach_id, date_str in predictions:
        result = predictor.predict_for_beach_date(
            beach_id=beach_id,
            forecast_date=date_str,
            model_name='LSTM'
        )
        
        if 'error' not in result:
            results.append({
                'Beach ID': beach_id,
                'Beach Name': result['beach_name'],
                'Date': date_str,
                'Probability': f"{result['percentage']:.1f}%",
                'Prediction': result['prediction'],
                'Confidence': result['confidence']
            })
    
    # Display results as table
    df = pd.DataFrame(results)
    print("\nPredictions:")
    print(df.to_string(index=False))
    
    # Compare models for one beach
    print("\nComparing all models for Beach 5 on 2025-03-15:")
    predictor.compare_predictions(beach_id=5, forecast_date='2025-03-15')

if __name__ == '__main__':
    main()
"""

# =============================================================================
# TYPICAL WORKFLOW
# =============================================================================

"""
1. Train models (one time):
   $ python train.py
   
   This takes 5-30 minutes depending on your hardware and generates:
   - baseline_model.pth
   - feedforward_model.pth
   - lstm_model.pth
   - gru_model.pth
   - conv1d_model.pth
   - hybrid_model.pth

2. Create prediction script:
   $ cat > predict_example.py << 'EOF'
   ... (copy code from above)
   EOF

3. Make predictions:
   $ python predict_example.py
   
   Output:
   Loading data...
   Loading models...
   ✓ Loaded LSTM model
   ✓ Loaded Hybrid model
   
   Making predictions...
   
   Predictions:
   Beach ID  Beach Name  Date        Probability  Prediction  Confidence
   5         Tel Aviv    2025-03-15  75.2%        Yes          High
   7         Netanya     2025-03-16  42.1%        No           Low
   10        Ashkelon    2025-03-17  68.9%        Yes          High

4. Or compare models interactively:
   $ python
   >>> from predictor import JellyfishPredictor
   >>> p = JellyfishPredictor()
   >>> p.load_data_cache()
   >>> p.load_model('LSTM', 'lstm_model.pth')
   >>> p.load_model('Hybrid', 'hybrid_model.pth')
   >>> p.compare_predictions(beach_id=5, forecast_date='2025-03-15')
"""

# =============================================================================
# KEY FEATURES
# =============================================================================

"""
✓ Clean separation of concerns:
  - models.py: Only model definitions
  - data_loader_forecasting.py: Only data loading
  - train.py: Only training logic
  - predictor.py: Only prediction interface

✓ Easy to use:
  - Load data once
  - Load models once
  - Make unlimited predictions
  - Compare models side-by-side

✓ Flexible:
  - Switch models without changing code
  - Add new models by inheriting from nn.Module
  - Use ensemble predictions
  - Get individual model outputs

✓ Production-ready:
  - Models saved and loaded from disk
  - Efficient batching
  - GPU support
  - Error handling
"""

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

"""
Q: Model file not found?
A: Make sure you ran 'python train.py' first to generate model weights

Q: Out of memory?
A: Reduce BATCH_SIZE in train.py (try 16 instead of 32)

Q: Predictions don't match expectations?
A: Check that beach_id and forecast_date are in the metadata:
   predictor.data_cache['metadata'][['beach_id', 'forecast_date']].head()

Q: Want to use a different model architecture?
A: Add your model to models.py, train it with train.py, load it with predictor

Q: Want to retrain a specific model?
A: Edit train.py to comment out models you don't want, then run again
"""

print(__doc__)
