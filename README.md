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
python scripts/train.py
```

4. Run prediction example:

```bash
python scripts/predict.py
```

5. Run model evaluation:

```bash
python scripts/evaluate.py
```

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
