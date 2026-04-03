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

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train models:

```bash
python scripts/train.py
```

3. Run prediction example:

```bash
python scripts/predict.py
```

4. Run model evaluation:

```bash
python scripts/evaluate.py
```

## Imports

You can now use package-style imports:

```python
from jellyfish.data_loader import load_jellyfish_data
from jellyfish.models import HybridNet
from jellyfish.predictor import JellyfishPredictor
from jellyfish.weather import IMSWeatherFetcher
```
