#!/usr/bin/env python3
"""Entrypoint wrapper for prediction examples."""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jellyfish.predict_example import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrypoint wrapper for jellyfish prediction example")
    parser.add_argument('--days-ahead', type=int, default=None)
    parser.add_argument('--beach-id', type=int, default=None)
    parser.add_argument('--lookback-days', type=int, default=7)
    parser.add_argument('--use-integrated-data', action='store_true')
    parser.add_argument('--weather-csv-path', type=str, default='data/IMS/data_202603142120.csv')
    parser.add_argument('--disable-live-xml', action='store_true')
    args = parser.parse_args()

    main(
        days_ahead=args.days_ahead,
        beach_id=args.beach_id,
        lookback_days=args.lookback_days,
        use_integrated_data=args.use_integrated_data,
        weather_csv_path=args.weather_csv_path,
        include_live_xml=not args.disable_live_xml,
    )
