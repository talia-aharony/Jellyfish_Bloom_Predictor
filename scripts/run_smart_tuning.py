#!/usr/bin/env python3
"""Smart hyperparameter tuning runner - 150 runs based on previous best configurations."""

import json
import subprocess
from pathlib import Path
from datetime import datetime

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(f'reports/smart_tuning_{timestamp}')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    transcript_path = run_dir / f'terminal_output_{timestamp}.log'
    manifest_path = run_dir / f'manifest_{timestamp}.json'
    
    # Load configurations
    with open('scripts/smart_tuning_configs.json') as f:
        config_data = json.load(f)
    configs = config_data['configs']
    
    # Header
    header = f"""====================================================================================
SMART HYPERPARAMETER TUNING - {timestamp}
====================================================================================
Strategy: Stratified sampling (60% high-confidence, 30% exploration, 10% edge cases)
Total configurations: {len(configs)}
Objective: Maximize precision/F1 while keeping recall >= 0.95
Output dir: {run_dir}
Transcript: {transcript_path}
Base config: 12 epochs, patience=8, threshold_target_recall=0.95
All runs: integrated data + live XML
====================================================================================
"""
    
    print(header)
    
    results = []
    with open(transcript_path, 'w') as log:
        log.write(header + '\n')
        
        for idx, config in enumerate(configs, 1):
            run_name = f"st{idx:03d}"
            report_file = run_dir / f"{run_name}.json"
            
            # Build command with correct CLI argument names (all use hyphens)
            cmd = [
                'python3', 'jellyfish/train.py',
                '--use-integrated-data',
                '--lookback-days', str(config['lookback_days']),
                '--hybrid-hidden-dim', str(config['hidden_dim']),
                '--dropout-prob', str(config['dropout']),
                '--learning-rate', str(config['lr']),
                '--batch-size', str(config['batch_size']),
                '--num-epochs', '12',
                '--patience', '8',
                '--positive-class-weight', str(config['positive_class_weight']),
                '--scheduler-factor', str(config['scheduler_factor']),
                '--scheduler-patience', '1',
                '--threshold-target-recall', '0.95',
                '--report-path', str(report_file),
            ]
            
            msg = (f"\n[{idx:3d}/{len(configs)}] {run_name}: "
                   f"lb={config['lookback_days']:2d} hd={config['hidden_dim']:3d} "
                   f"do={config['dropout']:.2f} lr={config['lr']:.6f} "
                   f"bs={config['batch_size']:2d} pw={config['positive_class_weight']:.2f}")
            print(msg, flush=True)
            log.write(msg + '\n')
            log.flush()
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                
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
                    msg2 = (f"   ✓ recall={metrics.get('recall', 0):.4f} "
                           f"precision={metrics.get('precision', 0):.4f} "
                           f"f1={metrics.get('f1', 0):.4f} "
                           f"auc={metrics.get('auc', 0):.4f}")
                    print(msg2, flush=True)
                    log.write(msg2 + '\n')
                else:
                    msg2 = f"   ✗ Failed (code={result.returncode})"
                    print(msg2, flush=True)
                    log.write(msg2 + '\n')
                    if result.stderr:
                        err_msg = f"   Err: {result.stderr[:100]}"
                        print(err_msg, flush=True)
                        log.write(err_msg + '\n')
            except subprocess.TimeoutExpired:
                print(f"   ✗ Timeout (3600s)", flush=True)
                log.write(f"   ✗ Timeout (3600s)\n")
            except Exception as e:
                msg2 = f"   ✗ Exception: {e}"
                print(msg2, flush=True)
                log.write(msg2 + '\n')
        
        # Summary section in log
        log.write(f"\n{'='*60}\n")
        log.write(f"Run Summary\n")
        log.write(f"{'='*60}\n")
        log.write(f"Total configs: {len(configs)}\n")
        log.write(f"Successful runs: {len(results)}\n")
    
    # Find best under recall constraint
    best = None
    for r in results:
        recall = r['metrics'].get('recall', 0)
        if recall >= 0.95:
            if best is None or r['metrics'].get('precision', 0) > best['metrics'].get('precision', 0):
                best = r
    
    # Generate manifest
    manifest = {
        'timestamp': timestamp,
        'total_configs': len(configs),
        'successful_runs': len(results),
        'objective': 'maximize precision/f1 while keeping recall >= 0.95',
        'runs': results,
        'best_under_recall_constraint': best,
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Final summary
    footer = f"""
{'='*60}
SMART TUNING COMPLETE - {timestamp}
{'='*60}
Completed {len(results)}/{len(configs)} runs successfully
Run directory: {run_dir}
Transcript:   {transcript_path}
Manifest:     {manifest_path}
{'='*60}
"""
    print(footer)
    
    if best:
        print(f"\nBest under recall>=0.95 constraint: {best['run']}")
        print(f"  Config: lb={best['config']['lookback_days']} hd={best['config']['hidden_dim']} "
              f"do={best['config']['dropout']} lr={best['config']['lr']} bs={best['config']['batch_size']} "
              f"pw={best['config']['positive_class_weight']}")
        print(f"  Metrics: recall={best['metrics'].get('recall', 0):.4f} "
              f"precision={best['metrics'].get('precision', 0):.4f} "
              f"f1={best['metrics'].get('f1', 0):.4f} "
              f"auc={best['metrics'].get('auc', 0):.4f}")

if __name__ == '__main__':
    main()
