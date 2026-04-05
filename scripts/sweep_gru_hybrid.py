#!/usr/bin/env python3
"""Run large GRU/Hybrid hyperparameter sweeps and save one report per run.

This script calls scripts/train.py many times with different settings.
Use scripts/compare_reports.py afterward to rank runs.
"""

import argparse
import itertools
import json
import os
import random
import time
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


PRESETS = {
    "focused": {
        "lookback_days": [14, 21],
        "learning_rates": [0.001, 0.0005],
        "dropouts": [0.2],
        "hybrid_hidden_dims": [96],
        "batch_sizes": [32],
        "epoch_options": [80],
        "patiences": [10],
    },
    "quick": {
        "lookback_days": [DEFAULT_LOOKBACK_DAYS],
        "learning_rates": [0.001, 0.0005],
        "dropouts": [0.2, 0.3],
        "hybrid_hidden_dims": [48, 64],
        "batch_sizes": [DEFAULT_BATCH_SIZE],
        "epoch_options": [DEFAULT_NUM_EPOCHS],
        "patiences": [DEFAULT_PATIENCE],
    },
    "standard": {
        "lookback_days": [7, 14, 21],
        "learning_rates": [0.001, 0.0007, 0.0005, 0.0003],
        "dropouts": [0.15, 0.2, 0.25, 0.3],
        "hybrid_hidden_dims": [32, 48, 64, 96],
        "batch_sizes": [16, 32],
        "epoch_options": [80, 100],
        "patiences": [10, 15],
    },
    "large": {
        "lookback_days": [7, 10, 14, 21, 28],
        "learning_rates": [0.001, 0.0008, 0.0006, 0.0005, 0.0003, 0.0002],
        "dropouts": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        "hybrid_hidden_dims": [24, 32, 48, 64, 80, 96],
        "batch_sizes": [16, 24, 32, 48],
        "epoch_options": [60, 80, 100, 120],
        "patiences": [8, 10, 12, 15],
    },
}


def parse_float_list(value: str):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_list(value: str):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def dedupe_keep_order(values):
    out = []
    seen = set()
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def main():
    parser = argparse.ArgumentParser(description="Large GRU/Hybrid sweep runner")
    parser.add_argument("--preset", type=str, default="focused", choices=sorted(PRESETS.keys()))
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Single lookback override")
    parser.add_argument("--lookback-days-list", type=str, default="", help="Comma list override, e.g. 7,14,21")
    parser.add_argument("--weather-csv-path", type=str, default=DEFAULT_WEATHER_CSV_PATH)
    parser.add_argument("--use-integrated-data", action="store_true", default=DEFAULT_USE_INTEGRATED_DATA)
    parser.add_argument("--disable-live-xml", action="store_true", default=not DEFAULT_INCLUDE_LIVE_XML)

    parser.add_argument("--learning-rates", type=str, default="", help="Comma list override")
    parser.add_argument("--dropouts", type=str, default="", help="Comma list override")
    parser.add_argument("--hybrid-hidden-dims", type=str, default="", help="Comma list override")
    parser.add_argument("--batch-sizes", type=str, default="", help="Comma list override")
    parser.add_argument("--epoch-options", type=str, default="", help="Comma list override")
    parser.add_argument("--patiences", type=str, default="", help="Comma list override")

    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)

    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--tag", type=str, default="sweep")
    parser.add_argument("--models", type=str, default="GRU,Hybrid", help="Comma-separated models to train")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run all combinations")
    parser.add_argument("--sample-runs", type=int, default=0, help="Randomly sample this many runs from full grid (0=all)")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle run order")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with existing report file")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue sweep even if one run fails")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Sleep between runs")
    parser.add_argument("--no-timestamp", action="store_true", help="Use stable report names for resume workflows")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    args = parser.parse_args()

    randomizer = random.Random(args.seed)

    preset = PRESETS[args.preset]

    lookback_days_list = preset["lookback_days"]
    learning_rates = preset["learning_rates"]
    dropouts = preset["dropouts"]
    hidden_dims = preset["hybrid_hidden_dims"]
    batch_sizes = preset["batch_sizes"]
    epoch_options = preset["epoch_options"]
    patiences = preset["patiences"]

    if args.lookback_days_list:
        lookback_days_list = parse_int_list(args.lookback_days_list)
    elif args.lookback_days not in lookback_days_list:
        lookback_days_list = [args.lookback_days]

    if args.learning_rates:
        learning_rates = parse_float_list(args.learning_rates)
    if args.dropouts:
        dropouts = parse_float_list(args.dropouts)
    if args.hybrid_hidden_dims:
        hidden_dims = parse_int_list(args.hybrid_hidden_dims)
    if args.batch_sizes:
        batch_sizes = parse_int_list(args.batch_sizes)
    if args.epoch_options:
        epoch_options = parse_int_list(args.epoch_options)
    if args.patiences:
        patiences = parse_int_list(args.patiences)

    # Fallback single-value overrides if user did not pass list overrides
    if not args.batch_sizes and args.batch_size != DEFAULT_BATCH_SIZE:
        batch_sizes = [args.batch_size]
    if not args.epoch_options and args.num_epochs != DEFAULT_NUM_EPOCHS:
        epoch_options = [args.num_epochs]
    if not args.patiences and args.patience != DEFAULT_PATIENCE:
        patiences = [args.patience]

    lookback_days_list = dedupe_keep_order(lookback_days_list)
    learning_rates = dedupe_keep_order(learning_rates)
    dropouts = dedupe_keep_order(dropouts)
    hidden_dims = dedupe_keep_order(hidden_dims)
    batch_sizes = dedupe_keep_order(batch_sizes)
    epoch_options = dedupe_keep_order(epoch_options)
    patiences = dedupe_keep_order(patiences)

    os.makedirs(args.reports_dir, exist_ok=True)

    combos = list(
        itertools.product(
            lookback_days_list,
            learning_rates,
            dropouts,
            hidden_dims,
            batch_sizes,
            epoch_options,
            patiences,
        )
    )

    if args.shuffle:
        randomizer.shuffle(combos)

    if args.sample_runs > 0 and args.sample_runs < len(combos):
        combos = randomizer.sample(combos, args.sample_runs)

    if args.max_runs > 0:
        combos = combos[: args.max_runs]

    if not combos:
        print("No hyperparameter combinations to run.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": args.preset,
        "seed": args.seed,
        "use_integrated_data": bool(args.use_integrated_data),
        "include_live_xml": not bool(args.disable_live_xml),
        "weather_csv_path": args.weather_csv_path,
        "models": args.models,
        "grid": {
            "lookback_days": lookback_days_list,
            "learning_rates": learning_rates,
            "dropouts": dropouts,
            "hybrid_hidden_dims": hidden_dims,
            "batch_sizes": batch_sizes,
            "epoch_options": epoch_options,
            "patiences": patiences,
        },
        "planned_runs": len(combos),
    }
    manifest_path = os.path.join(args.reports_dir, f"{args.tag}_{timestamp}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Total planned runs: {len(combos)}")
    print(f"Reports directory: {args.reports_dir}")
    print(f"Manifest: {manifest_path}")
    print()

    num_failed = 0
    num_skipped = 0
    num_completed = 0
    cancelled = False

    for run_idx, (lb, lr, dropout, hdim, bsz, epochs, pat) in enumerate(combos, start=1):
        prefix = f"{args.tag}_"
        if not args.no_timestamp:
            prefix += f"{timestamp}_"

        report_name = (
            f"{prefix}run{run_idx:04d}_"
            f"lb{lb}_lr{lr}_do{dropout}_hd{hdim}_bs{bsz}_ep{epochs}_pt{pat}.json"
        )
        report_path = os.path.join(args.reports_dir, report_name)

        if args.skip_existing and os.path.exists(report_path):
            print(f"[{run_idx}/{len(combos)}] SKIP existing report: {report_path}")
            num_skipped += 1
            continue

        cmd = [
            sys.executable,
            "scripts/train.py",
            "--lookback-days", str(lb),
            "--weather-csv-path", args.weather_csv_path,
            "--batch-size", str(bsz),
            "--learning-rate", str(lr),
            "--dropout-prob", str(dropout),
            "--num-epochs", str(epochs),
            "--patience", str(pat),
            "--hybrid-hidden-dim", str(hdim),
            "--models", args.models,
            "--report-path", report_path,
        ]

        if args.use_integrated_data:
            cmd.append("--use-integrated-data")
        if args.disable_live_xml:
            cmd.append("--disable-live-xml")

        print(f"[{run_idx}/{len(combos)}] {' '.join(cmd)}")

        if args.dry_run:
            continue

        try:
            completed = subprocess.run(cmd, cwd=ROOT)
        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C). Stopping sweep cleanly.")
            cancelled = True
            break

        if completed.returncode != 0:
            print(f"Run failed with code {completed.returncode}")
            num_failed += 1
            if not args.continue_on_error:
                print("Stopping sweep due to failure.")
                break
        else:
            num_completed += 1

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    print()
    if cancelled:
        print("Sweep cancelled by user.")
    else:
        print("Sweep finished.")
    print(f"Completed runs: {num_completed}")
    print(f"Failed runs: {num_failed}")
    print(f"Skipped runs: {num_skipped}")
    print("Next: rank reports with:")
    print("  python scripts/compare_reports.py --pattern \"reports/*.json\" --model Hybrid --sort-by f1")


if __name__ == "__main__":
    main()
