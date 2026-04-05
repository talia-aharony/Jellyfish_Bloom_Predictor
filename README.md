# Jellyfish Bloom Predictor

Forecast jellyfish presence at Israeli Mediterranean beaches using citizen-science observations and PyTorch models.

## Project Layout

```
Jellyfish_Bloom_Predictor/
├── jellyfish/                  # Core package
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_loader_forecasting.py
│   ├── evaluator.py
│   ├── evaluate_models.py
│   ├── models.py
│   ├── predict_example.py
│   ├── predictor.py
│   ├── train.py
│   └── weather.py
├── scripts/                    # Stable entrypoints
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── data/                       # Raw data files
├── README.md
└── requirements.txt
```

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Train models:

```bash
python scripts/train.py --lookback-days 7
```

Train with all input sources (citizen science + IMS weather CSV + live RSS):

```bash
python scripts/train.py \
	--use-integrated-data \
	--weather-csv-path data/IMS/data_202603142120.csv \
	--lookback-days 14
```

4. Run prediction example:

```bash
python scripts/predict.py --lookback-days 7
```

Predict using integrated data inputs (including live RSS where available):

```bash
python scripts/predict.py \
	--use-integrated-data \
	--weather-csv-path data/IMS/data_202603142120.csv \
	--lookback-days 14 \
	--beach-id 14 \
	--days-ahead 2
```

Note on live RSS horizon:
- RSS city/sea/alert forecasts are only available for the next few days.
- The predictor applies RSS values only for dates present in live RSS feeds.
- For dates beyond RSS coverage, it falls back to sequence carry-forward/trend logic.

5. Run model evaluation:

```bash
python scripts/evaluate.py
```

## Training + Hyperparameter Tuning Notes

Central defaults are now in [jellyfish/settings.py](jellyfish/settings.py).
This is the single source for default hyperparameters used by training, prediction, and evaluation entrypoints.
If you want to tune in one place (for example only `lookback_days`), edit that file and rerun scripts.

Use the direct training script when tuning hyperparameters:

```bash
python jellyfish/train.py \
	--lookback-days 7 \
	--batch-size 32 \
	--learning-rate 0.001 \
	--dropout-prob 0.35 \
	--num-epochs 100 \
	--patience 15 \
	--hybrid-hidden-dim 96 \
	--report-path training_report_latest.json
```

Each run now saves a JSON report containing:
- Training configuration (`lookback_days`, `batch_size`, `learning_rate`, `dropout_prob`, `num_epochs`, `patience`, `hybrid_hidden_dim`)
- Model metrics (`accuracy`, `precision`, `recall`, `f1`, `auc`, confusion matrix)

### Suggested experiment grid for HybridNet

Run separate experiments by varying one factor at a time:

1. Learning rate: `0.001`, `0.0005`, `0.0003`
2. Dropout: `0.30`, `0.35`, `0.40`
3. Hidden dimension: `64`, `96`, `128`
4. Batch size: `16`, `32`, `64`

Example run names:

```bash
python jellyfish/train.py --learning-rate 0.0005 --dropout-prob 0.35 --hybrid-hidden-dim 128 --report-path reports/hybrid_lr5e4_do35_hd128.json
```

Lookback sweep example:

```bash
python jellyfish/train.py --lookback-days 7  --report-path reports/hybrid_lb7.json
python jellyfish/train.py --lookback-days 14 --report-path reports/hybrid_lb14.json
python jellyfish/train.py --lookback-days 21 --report-path reports/hybrid_lb21.json
```

### How to comment on tuning in your report

For each run, summarize:
- What changed (hyperparameter delta)
- Effect on `F1` and `AUC` (primary quality metrics)
- Effect on precision/recall balance
- Final decision and why (best tradeoff)

### Rank all experiment reports automatically

Use this helper to compare saved JSON reports and rank runs by your chosen metric:

```bash
python scripts/compare_reports.py --model Hybrid --sort-by f1
```

Useful variants:

```bash
python scripts/compare_reports.py --pattern "training_report*.json" --extra-pattern "reports/*.json" --model Hybrid --sort-by auc --top-k 10
python scripts/compare_reports.py --model LSTM --sort-by f1
```

This makes your hyperparameter commentary easier: cite the top-ranked run and compare it against the next 2-3 runs by F1/AUC and precision/recall tradeoff.

## Direct Execution (Also Supported)

These package files can also be run directly:

```bash
python jellyfish/train.py
python jellyfish/predict_example.py
python jellyfish/evaluate_models.py
```

## Imports

You can now use package-style imports:

```python
from jellyfish.data_loader import load_jellyfish_data
from jellyfish.models import HybridNet
from jellyfish.predictor import JellyfishPredictor
from jellyfish.weather import IMSWeatherFetcher
```
