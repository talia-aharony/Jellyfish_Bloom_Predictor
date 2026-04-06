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
    DEFAULT_SWEEP_PRESET,
    DEFAULT_SWEEP_MODELS,
    SWEEP_PRESET_FOCUSED,
    SWEEP_PRESET_JELLYNET_STRONG,
    SWEEP_PRESET_QUICK,
    SWEEP_PRESET_STANDARD,
    SWEEP_PRESET_LARGE,
)


PRESETS = {
    "focused": SWEEP_PRESET_FOCUSED,
    "jellynet_strong": SWEEP_PRESET_JELLYNET_STRONG,
    "quick": SWEEP_PRESET_QUICK,
    "standard": SWEEP_PRESET_STANDARD,
    "large": SWEEP_PRESET_LARGE,
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


def read_report_metric(report_path: str, model_name: str, metric_name: str):
    """Read objective metric from a saved training report JSON."""
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None

    results = payload.get("results", {})
    if not isinstance(results, dict) or not results:
        return None

    metrics = results.get(model_name)
    if metrics is None:
        # Fallback to first available model in report
        first_key = next(iter(results.keys()))
        metrics = results[first_key]

    if not isinstance(metrics, dict):
        return None

    value = metrics.get(metric_name)
    try:
        return float(value)
    except Exception:
        return None


def all_grid_combos(lookback_days_list, learning_rates, dropouts, hidden_dims, batch_sizes, epoch_options, patiences):
    return list(
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


def adaptive_combo_sequence(combos, max_runs):
    """Direction-aware coordinate search over a fixed grid of combinations.

    Returns a sequence of combo indices to evaluate.
    """
    if not combos:
        return []

    # Build per-dimension value lists from the full combo grid
    dims = list(zip(*combos))
    values_per_dim = [sorted(set(d)) for d in dims]
    combo_to_idx = {c: i for i, c in enumerate(combos)}

    # Start at the median value along each dimension
    current_point = tuple(vals[len(vals) // 2] for vals in values_per_dim)
    if current_point not in combo_to_idx:
        current_point = combos[0]

    visited_combo_indices = []
    tried_points = set()
    tried_points.add(current_point)
    visited_combo_indices.append(combo_to_idx[current_point])

    directions = {dim: 1 for dim in range(len(values_per_dim))}  # +1 higher, -1 lower
    param_cursor = 0

    # The caller will update these via feedback at runtime
    state = {
        "current_point": current_point,
        "directions": directions,
        "param_cursor": param_cursor,
        "tried_points": tried_points,
    }

    return visited_combo_indices, state, values_per_dim, combo_to_idx


def adaptive_next_point(state, values_per_dim):
    """Propose the next nearby point to try, preferring current improvement directions."""
    current_point = state["current_point"]
    directions = state["directions"]
    tried_points = state["tried_points"]
    n_dims = len(values_per_dim)

    for offset in range(n_dims):
        dim = (state["param_cursor"] + offset) % n_dims
        dim_values = values_per_dim[dim]
        cur_val = current_point[dim]
        cur_idx = dim_values.index(cur_val)

        preferred = directions[dim]
        candidate_dirs = [preferred, -preferred]
        for step_dir in candidate_dirs:
            nxt_idx = cur_idx + step_dir
            if 0 <= nxt_idx < len(dim_values):
                candidate = list(current_point)
                candidate[dim] = dim_values[nxt_idx]
                candidate = tuple(candidate)
                if candidate not in tried_points:
                    state["param_cursor"] = (dim + 1) % n_dims
                    return candidate, dim, step_dir

    return None, None, None


def adaptive_fallback_untried(combos, state):
    """Return any untried point if local neighbors are exhausted."""
    tried_points = state["tried_points"]
    for c in combos:
        if c not in tried_points:
            return c
    return None


def combo_to_parts(combo):
    lb, lr, dropout, hdim, bsz, epochs, pat = combo
    return lb, lr, dropout, hdim, bsz, epochs, pat


def main():
    parser = argparse.ArgumentParser(description="Large GRU/Hybrid sweep runner")
    parser.add_argument("--preset", type=str, default=DEFAULT_SWEEP_PRESET, choices=sorted(PRESETS.keys()))
    parser.add_argument("--search-mode", type=str, default="grid", choices=["grid", "adaptive"], help="Sweep strategy")
    parser.add_argument("--sort-by", type=str, default="f1", choices=["f1", "auc", "accuracy", "precision", "recall", "val_best_f1"], help="Objective metric for adaptive search")
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
    parser.add_argument("--models", type=str, default=DEFAULT_SWEEP_MODELS, help="Comma-separated models to train")
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

    combos = all_grid_combos(
        lookback_days_list,
        learning_rates,
        dropouts,
        hidden_dims,
        batch_sizes,
        epoch_options,
        patiences,
    )

    if args.shuffle:
        randomizer.shuffle(combos)

    if args.sample_runs > 0 and args.sample_runs < len(combos):
        combos = randomizer.sample(combos, args.sample_runs)

    if not combos:
        print("No hyperparameter combinations to run.")
        return

    planned_runs = len(combos) if args.max_runs <= 0 else min(args.max_runs, len(combos))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": args.preset,
        "seed": args.seed,
        "use_integrated_data": bool(args.use_integrated_data),
        "include_live_xml": not bool(args.disable_live_xml),
        "weather_csv_path": args.weather_csv_path,
        "models": args.models,
        "search_mode": args.search_mode,
        "sort_by": args.sort_by,
        "grid": {
            "lookback_days": lookback_days_list,
            "learning_rates": learning_rates,
            "dropouts": dropouts,
            "hybrid_hidden_dims": hidden_dims,
            "batch_sizes": batch_sizes,
            "epoch_options": epoch_options,
            "patiences": patiences,
        },
        "model_output_root": "models",
        "planned_runs": planned_runs,
    }
    manifest_path = os.path.join(args.reports_dir, f"{args.tag}_{timestamp}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Total planned runs: {planned_runs}")
    print(f"Reports directory: {args.reports_dir}")
    print(f"Manifest: {manifest_path}")
    print()

    num_failed = 0
    num_skipped = 0
    num_completed = 0
    cancelled = False

    run_model_names = [m.strip() for m in args.models.split(',') if m.strip()]
    objective_model = run_model_names[-1] if run_model_names else 'JellyfishNet'

    adaptive_state = None
    values_per_dim = None
    combo_to_index_map = None
    adaptive_last_change = None
    best_score = None
    best_combo = None

    if args.search_mode == "adaptive":
        seed_indices, adaptive_state, values_per_dim, combo_to_index_map = adaptive_combo_sequence(combos, planned_runs)
        pending_combo_indices = list(seed_indices)
    else:
        if args.max_runs > 0:
            combos = combos[: args.max_runs]
        pending_combo_indices = list(range(len(combos)))

    run_idx = 0
    while pending_combo_indices and run_idx < planned_runs:
        combo_idx = pending_combo_indices.pop(0)
        lb, lr, dropout, hdim, bsz, epochs, pat = combo_to_parts(combos[combo_idx])
        run_idx += 1
        prefix = f"{args.tag}_"
        if not args.no_timestamp:
            prefix += f"{timestamp}_"

        run_stem = (
            f"{prefix}run{run_idx:04d}_"
            f"lb{lb}_lr{lr}_do{dropout}_hd{hdim}_bs{bsz}_ep{epochs}_pt{pat}"
        )
        report_name = f"{run_stem}.json"
        report_path = os.path.join(args.reports_dir, report_name)
        model_output_dir = os.path.join("models", run_stem)

        if args.skip_existing and os.path.exists(report_path):
            print(f"[{run_idx}/{planned_runs}] SKIP existing report: {report_path}")
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
            "--output-dir", model_output_dir,
            "--report-path", report_path,
        ]

        if args.use_integrated_data:
            cmd.append("--use-integrated-data")
        if args.disable_live_xml:
            cmd.append("--disable-live-xml")

        print(f"[{run_idx}/{planned_runs}] {' '.join(cmd)}")

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

            if args.search_mode == "adaptive":
                score = read_report_metric(report_path, objective_model, args.sort_by)
                if score is not None:
                    improved = best_score is None or score > best_score
                    if improved:
                        best_score = score
                        best_combo = combos[combo_idx]
                        if adaptive_last_change is not None:
                            changed_dim, changed_step = adaptive_last_change
                            adaptive_state["directions"][changed_dim] = changed_step
                        adaptive_state["current_point"] = combos[combo_idx]
                    else:
                        if adaptive_last_change is not None:
                            changed_dim, changed_step = adaptive_last_change
                            adaptive_state["directions"][changed_dim] = -changed_step

                    print(
                        f"  [adaptive] {args.sort_by}={score:.4f}"
                        f" | best={best_score:.4f}"
                    )

                # Choose next point adaptively
                next_point, changed_dim, changed_step = adaptive_next_point(adaptive_state, values_per_dim)
                if next_point is None:
                    next_point = adaptive_fallback_untried(combos, adaptive_state)
                    adaptive_last_change = None
                else:
                    adaptive_last_change = (changed_dim, changed_step)

                if next_point is not None:
                    adaptive_state["tried_points"].add(next_point)
                    next_idx = combo_to_index_map[next_point]
                    pending_combo_indices.append(next_idx)

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
    if args.search_mode == "adaptive" and best_combo is not None and best_score is not None:
        lb, lr, dropout, hdim, bsz, epochs, pat = combo_to_parts(best_combo)
        print(
            f"Best adaptive run ({args.sort_by}={best_score:.4f}): "
            f"lb={lb}, lr={lr}, do={dropout}, hd={hdim}, bs={bsz}, ep={epochs}, pt={pat}"
        )
    rank_model = [m.strip() for m in args.models.split(',') if m.strip()]
    rank_model = rank_model[-1] if rank_model else 'JellyfishNet'
    print("Next: rank reports with:")
    print(f"  python scripts/compare_reports.py --pattern \"reports/*.json\" --model {rank_model} --sort-by f1")


if __name__ == "__main__":
    main()
