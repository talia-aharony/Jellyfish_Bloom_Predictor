#!/usr/bin/env python3
"""Entrypoint wrapper for model training."""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jellyfish.train import train_all_models, finetune_per_beach
from jellyfish.settings import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_WEATHER_CSV_PATH,
    DEFAULT_USE_INTEGRATED_DATA,
    DEFAULT_INCLUDE_LIVE_XML,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DROPOUT_PROB,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_HYBRID_HIDDEN_DIM,
    DEFAULT_REPORT_PATH,
    DEFAULT_MODEL_NAMES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_FINETUNE_LR,
    DEFAULT_MIN_SAMPLES_PER_BEACH,
    DEFAULT_LR_SCHEDULER_FACTOR,
    DEFAULT_LR_SCHEDULER_PATIENCE,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_THRESHOLD_MIN,
    DEFAULT_THRESHOLD_MAX,
    DEFAULT_THRESHOLD_STEPS,
    DEFAULT_THRESHOLD_MIN_PRECISION,
    DEFAULT_THRESHOLD_TARGET_RECALL,
    DEFAULT_POSITIVE_CLASS_WEIGHT,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrypoint wrapper for jellyfish training")
    parser.add_argument('--lookback-days', type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument('--use-integrated-data', action='store_true', default=DEFAULT_USE_INTEGRATED_DATA)
    parser.add_argument('--weather-csv-path', type=str, default=DEFAULT_WEATHER_CSV_PATH)
    parser.add_argument('--disable-live-xml', action='store_true', default=not DEFAULT_INCLUDE_LIVE_XML)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--dropout-prob', type=float, default=DEFAULT_DROPOUT_PROB)
    parser.add_argument('--num-epochs', type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    parser.add_argument('--hybrid-hidden-dim', type=int, default=DEFAULT_HYBRID_HIDDEN_DIM)
    parser.add_argument('--models', type=str, default=DEFAULT_MODEL_NAMES)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--report-path', type=str, default=DEFAULT_REPORT_PATH)
    parser.add_argument('--scheduler-factor', type=float, default=DEFAULT_LR_SCHEDULER_FACTOR)
    parser.add_argument('--scheduler-patience', type=int, default=DEFAULT_LR_SCHEDULER_PATIENCE)
    parser.add_argument('--grad-clip-norm', type=float, default=DEFAULT_GRAD_CLIP_NORM)
    parser.add_argument('--threshold-min', type=float, default=DEFAULT_THRESHOLD_MIN)
    parser.add_argument('--threshold-max', type=float, default=DEFAULT_THRESHOLD_MAX)
    parser.add_argument('--threshold-steps', type=int, default=DEFAULT_THRESHOLD_STEPS)
    parser.add_argument('--threshold-min-precision', type=float, default=DEFAULT_THRESHOLD_MIN_PRECISION)
    parser.add_argument('--threshold-target-recall', type=float, default=DEFAULT_THRESHOLD_TARGET_RECALL)
    parser.add_argument('--positive-class-weight', type=float, default=DEFAULT_POSITIVE_CLASS_WEIGHT)

    parser.add_argument('--finetune-per-beach', action='store_true')
    parser.add_argument('--global-checkpoint', type=str, default=None)
    parser.add_argument('--finetune-epochs', type=int, default=DEFAULT_FINETUNE_EPOCHS)
    parser.add_argument('--finetune-lr', type=float, default=DEFAULT_FINETUNE_LR)
    parser.add_argument('--min-samples', type=int, default=DEFAULT_MIN_SAMPLES_PER_BEACH)
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(',') if m.strip()]

    if args.finetune_per_beach:
        checkpoint = args.global_checkpoint or os.path.join(args.output_dir, 'jellyfishnet_model.pth')
        finetune_per_beach(
            global_checkpoint=checkpoint,
            lookback_days=args.lookback_days,
            use_integrated_data=args.use_integrated_data,
            weather_csv_path=args.weather_csv_path,
            include_live_xml=not args.disable_live_xml,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
            dropout_prob=args.dropout_prob,
            hybrid_hidden_dim=args.hybrid_hidden_dim,
            min_samples=args.min_samples,
            output_dir=args.output_dir,
            report_path='training_report_per_beach.json',
            scheduler_factor=args.scheduler_factor,
            scheduler_patience=args.scheduler_patience,
            grad_clip_norm=args.grad_clip_norm,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_steps=args.threshold_steps,
            threshold_min_precision=args.threshold_min_precision,
            threshold_target_recall=args.threshold_target_recall,
            positive_class_weight=args.positive_class_weight,
        )
    else:
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
            output_dir=args.output_dir,
            report_path=args.report_path,
            scheduler_factor=args.scheduler_factor,
            scheduler_patience=args.scheduler_patience,
            grad_clip_norm=args.grad_clip_norm,
            threshold_min=args.threshold_min,
            threshold_max=args.threshold_max,
            threshold_steps=args.threshold_steps,
            threshold_min_precision=args.threshold_min_precision,
            threshold_target_recall=args.threshold_target_recall,
            positive_class_weight=args.positive_class_weight,
        )
