#!/usr/bin/env python3
"""Entrypoint wrapper for model training."""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jellyfish.train import train_all_models
from jellyfish.settings import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_WEATHER_CSV_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DROPOUT_PROB,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_HYBRID_HIDDEN_DIM,
    DEFAULT_REPORT_PATH,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrypoint wrapper for jellyfish training")
    parser.add_argument('--lookback-days', type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument('--use-integrated-data', action='store_true')
    parser.add_argument('--weather-csv-path', type=str, default=DEFAULT_WEATHER_CSV_PATH)
    parser.add_argument('--disable-live-xml', action='store_true')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--dropout-prob', type=float, default=DEFAULT_DROPOUT_PROB)
    parser.add_argument('--num-epochs', type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    parser.add_argument('--hybrid-hidden-dim', type=int, default=DEFAULT_HYBRID_HIDDEN_DIM)
    parser.add_argument('--models', type=str, default='GRU,Hybrid')
    parser.add_argument('--report-path', type=str, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(',') if m.strip()]

    train_all_models(
        lookback_days=args.lookback_days,
        use_integrated_data=args.use_integrated_data,
        weather_csv_path=args.weather_csv_path,
        include_live_xml=not args.disable_live_xml,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_prob=args.dropout_prob,
        num_epochs=args.num_epochs,
        patience=args.patience,
        hybrid_hidden_dim=args.hybrid_hidden_dim,
        model_names=model_names,
        report_path=args.report_path,
    )
