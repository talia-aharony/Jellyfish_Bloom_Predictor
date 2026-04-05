#!/usr/bin/env python3
"""Run GRU/Hybrid hyperparameter sweep and save one report per run.

This script calls scripts/train.py multiple times with different settings.
Use scripts/compare_reports.py afterward to rank runs.
"""

import argparse
import itertools
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jellyfish.settings import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_WEATHER_CSV_PATH,
    DEFAULT_USE_INTEGRATED_DATA,
    DEFAULT_INCLUDE_LIVE_XML,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_PATIENCE,
)


def parse_float_list(value: str):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="GRU/Hybrid sweep runner")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--weather-csv-path", type=str, default=DEFAULT_WEATHER_CSV_PATH)
    parser.add_argument("--use-integrated-data", action="store_true", default=DEFAULT_USE_INTEGRATED_DATA)
    parser.add_argument("--disable-live-xml", action="store_true", default=not DEFAULT_INCLUDE_LIVE_XML)

    parser.add_argument("--learning-rates", type=str, default="0.001,0.0005,0.0003")
    parser.add_argument("--dropouts", type=str, default="0.20,0.25,0.30")
    parser.add_argument("--hybrid-hidden-dims", type=str, default="48,64,96")

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)

    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--tag", type=str, default="sweep")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run all combinations")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()

    learning_rates = parse_float_list(args.learning_rates)
    dropouts = parse_float_list(args.dropouts)
    hidden_dims = parse_int_list(args.hybrid_hidden_dims)

    os.makedirs(args.reports_dir, exist_ok=True)

    combos = list(itertools.product(learning_rates, dropouts, hidden_dims))
    if args.max_runs > 0:
        combos = combos[: args.max_runs]

    if not combos:
        print("No hyperparameter combinations to run.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Total planned runs: {len(combos)}")
    print(f"Reports directory: {args.reports_dir}")
    print()

    for run_idx, (lr, dropout, hdim) in enumerate(combos, start=1):
        report_name = (
            f"{args.tag}_{timestamp}_run{run_idx:02d}_"
            f"lb{args.lookback_days}_lr{lr}_do{dropout}_hd{hdim}.json"
        )
        report_path = os.path.join(args.reports_dir, report_name)

        cmd = [
            sys.executable,
            "scripts/train.py",
            "--lookback-days", str(args.lookback_days),
            "--weather-csv-path", args.weather_csv_path,
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(lr),
            "--dropout-prob", str(dropout),
            "--num-epochs", str(args.num_epochs),
            "--patience", str(args.patience),
            "--hybrid-hidden-dim", str(hdim),
            "--report-path", report_path,
        ]

        if args.use_integrated_data:
            cmd.append("--use-integrated-data")
        if args.disable_live_xml:
            cmd.append("--disable-live-xml")

        print(f"[{run_idx}/{len(combos)}] {' '.join(cmd)}")

        if args.dry_run:
            continue

        completed = subprocess.run(cmd, cwd=ROOT)
        if completed.returncode != 0:
            print(f"Run failed with code {completed.returncode}; stopping sweep.")
            return

    print()
    print("Sweep completed.")
    print("Next: rank reports with:")
    print("  python scripts/compare_reports.py --pattern \"reports/*.json\" --model Hybrid --sort-by f1")


if __name__ == "__main__":
    main()
