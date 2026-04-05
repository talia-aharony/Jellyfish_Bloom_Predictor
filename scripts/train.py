#!/usr/bin/env python3
"""Entrypoint wrapper for model training."""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jellyfish.train import train_all_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrypoint wrapper for jellyfish training")
    parser.add_argument('--lookback-days', type=int, default=7)
    parser.add_argument('--use-integrated-data', action='store_true')
    parser.add_argument('--weather-csv-path', type=str, default='data/IMS/data_202603142120.csv')
    parser.add_argument('--disable-live-xml', action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--dropout-prob', type=float, default=0.3)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--hybrid-hidden-dim', type=int, default=96)
    parser.add_argument('--report-path', type=str, default='training_report_latest.json')
    args = parser.parse_args()

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
        report_path=args.report_path,
    )
