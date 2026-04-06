#!/bin/bash
# Smart hyperparameter tuning runner - 150 runs based on previous best configs
# Logs all output to timestamped transcript file with manifests

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="reports/smart_tuning_${TIMESTAMP}"
TRANSCRIPT="${RUN_DIR}/terminal_output_${TIMESTAMP}.log"
MANIFEST="${RUN_DIR}/manifest_${TIMESTAMP}.json"

mkdir -p "$RUN_DIR"

echo "===================================================================================="
echo "SMART HYPERPARAMETER TUNING - $TIMESTAMP"
echo "===================================================================================="
echo "Strategy: Stratified sampling (60% high-confidence, 30% exploration, 10% edge cases)"
echo "Total runs: 150"
echo "Objective: Maximize precision/F1 while keeping recall >= 0.95"
echo "Output dir: $RUN_DIR"
echo "Transcript: $TRANSCRIPT"
echo "===================================================================================="
echo ""

# Generate configs if not already done
if [ ! -f "scripts/smart_tuning_configs.json" ]; then
    echo "Generating 150 configurations..."
    python3 scripts/smart_tuning_generator.py
fi

# Load configs into array
CONFIGS=$(python3 -c "import json; d=json.load(open('scripts/smart_tuning_configs.json')); print(json.dumps([c for c in d['configs']], separators=(',', ':')))")

# Extract config count
CONFIG_COUNT=$(python3 -c "import json; d=json.load(open('scripts/smart_tuning_configs.json')); print(d['total_configs'])")

echo "Loaded $CONFIG_COUNT configurations."
echo ""

# Prepare header info for logs
HEADER="SMART HYPERPARAMETER TUNING - $TIMESTAMP
========================================================
Total Configurations: $CONFIG_COUNT  
Strategy: Stratified sampling (60% HC, 30% exploration, 10% edges)
Objective: Maximize precision/F1 while keeping recall >= 0.95
Base epochs: 12
Scheduler patience: 8
Threshold target recall: 0.95
All runs use: integrated data + live XML
========================================================"

# Start logging
{
    echo "$HEADER"
    echo ""
    
    # Run each configuration
    python3 << 'PYEOF'
import json
import subprocess
import sys
from pathlib import Path

configs = json.load(open('scripts/smart_tuning_configs.json'))['configs']
        run_timestamp = sys.argv[1]
        run_dir = Path('reports/smart_tuning_' + run_timestamp)
results = []

for idx, config in enumerate(configs, 1):
    run_name = f"st{idx:03d}"
    report_file = run_dir / f"{run_name}.json"
    
    # Build command
    cmd = [
        'python3', 'jellyfish/train.py',
        '--use_integrated_data',
        '--include_live_xml',
        '--lookback_days', str(config['lookback_days']),
        '--hybrid_hidden_dim', str(config['hidden_dim']),
        '--dropout_prob', str(config['dropout']),
        '--learning_rate', str(config['lr']),
        '--batch_size', str(config['batch_size']),
        '--num_epochs', str(12),
        '--patience', str(8),
        '--positive_class_weight', str(config['positive_class_weight']),
        '--scheduler_factor', str(config['scheduler_factor']),
        '--scheduler_patience', str(1),
        '--threshold_target_recall', str(0.95),
        '--report', str(report_file),
    ]
    
    print(f"\n[{idx:3d}/{len(configs)}] {run_name}: lb={config['lookback_days']} hd={config['hidden_dim']} "
          f"do={config['dropout']:.2f} lr={config['lr']:.6f} bs={config['batch_size']} pw={config['positive_class_weight']:.2f}", 
          flush=True)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        # Parse report if successful
        if result.returncode == 0 and report_file.exists():
            with open(report_file) as f:
                report = json.load(f)
            metrics = report.get('metrics', {})
            result_entry = {
                'run': run_name,
                'report_path': str(report_file),
                'config': config,
                'metrics': metrics,
            }
            results.append(result_entry)
            print(f"   ✓ recall={metrics.get('recall', 0):.4f} precision={metrics.get('precision', 0):.4f} "
                  f"f1={metrics.get('f1', 0):.4f}", flush=True)
        else:
            print(f"   ✗ Failed or timeout", flush=True)
            if result.stderr:
                print(f"   Error: {result.stderr[:200]}", flush=True)
    except Exception as e:
        print(f"   ✗ Exception: {e}", flush=True)

# Find best under recall constraint
best = None
for r in results:
    if r['metrics'].get('recall', 0) >= 0.95:
        if best is None or r['metrics'].get('precision', 0) > best['metrics'].get('precision', 0):
            best = r

# Generate manifest
manifest = {
    'timestamp': sys.argv[1],
    'total_configs': len(configs),
    'successful_runs': len(results),
    'objective': 'maximize precision/f1 while keeping recall >= 0.95',
    'runs': results,
    'best_under_recall_constraint': best,
}

manifest_path = run_dir / f"manifest_{sys.argv[1]}.json"
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"\n{'='*60}", flush=True)
print(f"Completed {len(results)}/{len(configs)} runs successfully", flush=True)
if best:
    print(f"Best under recall>=0.95 constraint: {best['run']}", flush=True)
    print(f"  Config: lb={best['config']['lookback_days']} hd={best['config']['hidden_dim']} "
          f"do={best['config']['dropout']} lr={best['config']['lr']} bs={best['config']['batch_size']} "
          f"pw={best['config']['positive_class_weight']}", flush=True)
    print(f"  Metrics: recall={best['metrics'].get('recall', 0):.4f} precision={best['metrics'].get('precision', 0):.4f} "
          f"f1={best['metrics'].get('f1', 0):.4f} auc={best['metrics'].get('auc', 0):.4f}", flush=True)
print(f"Manifest: {manifest_path}", flush=True)

PYEOF

} 2>&1 | tee "$TRANSCRIPT"

# Summary
echo ""
echo "===================================================================================="
echo "SMART TUNING COMPLETE - $TIMESTAMP"
echo "===================================================================================="
echo "Run directory: $RUN_DIR"
echo "Transcript:   $TRANSCRIPT"
echo "Manifest:     $MANIFEST"
echo "===================================================================================="

exit 0
