#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from datetime import datetime

np.random.seed(42)

# 150 intelligently sampled configs
configs = []

# 60% high-confidence region (around best run: lb=21, hd=128, do=0.15, lr=0.0005, bs=16, pw=1.4, sf=0.3)
for i in range(90):
    c = {
        'lookback_days': max(14, min(28, int(np.random.normal(22, 3)))),
        'hidden_dim': max(64, min(256, int(np.random.choice([128, 160, 192, 224])))),
        'dropout': max(0.05, min(0.30, np.random.normal(0.13, 0.04))),
        'lr': max(0.0001, min(0.001, 10**np.random.normal(np.log10(0.0005), 0.3))),
        'batch_size': max(8, min(32, np.random.randint(12, 24))),
        'positive_class_weight': max(1.0, min(2.0, np.random.normal(1.5, 0.2))),
        'scheduler_factor': float(np.random.choice([0.2, 0.25, 0.3, 0.35])),
    }
    configs.append(c)

# 30% exploration region
for i in range(45):
    c = {
        'lookback_days': np.random.randint(14, 29),
        'hidden_dim': int(np.random.choice([64, 96, 128, 160, 192, 224, 256])),
        'dropout': np.random.uniform(0.05, 0.25),
        'lr': 10**np.random.uniform(np.log10(0.0002), np.log10(0.001)),
        'batch_size': int(np.random.choice([8, 12, 16, 20, 24, 32])),
        'positive_class_weight': np.random.uniform(1.0, 2.0),
        'scheduler_factor': float(np.random.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.4])),
    }
    configs.append(c)

# 10% edge probing
for i in range(15):
    choice = np.random.choice(['low_lb_small_hd', 'high_lb_large_hd', 'very_low_lr', 'high_lr', 'high_do', 'low_do'])
    if choice == 'low_lb_small_hd':
        c = {
            'lookback_days': np.random.randint(14, 17),
            'hidden_dim': int(np.random.choice([64, 96])),
            'dropout': np.random.uniform(0.05, 0.12),
            'lr': np.random.uniform(0.0006, 0.001),
            'batch_size': int(np.random.choice([24, 32])),
            'positive_class_weight': np.random.uniform(1.4, 1.8),
            'scheduler_factor': 0.2,
        }
    elif choice == 'high_lb_large_hd':
        c = {
            'lookback_days': np.random.randint(25, 29),
            'hidden_dim': int(np.random.choice([224, 256])),
            'dropout': np.random.uniform(0.10, 0.20),
            'lr': np.random.uniform(0.0002, 0.0004),
            'batch_size': int(np.random.choice([8, 12])),
            'positive_class_weight': np.random.uniform(1.0, 1.4),
            'scheduler_factor': 0.3,
        }
    elif choice == 'very_low_lr':
        c = {
            'lookback_days': np.random.randint(18, 24),
            'hidden_dim': int(np.random.choice([128, 160, 192])),
            'dropout': np.random.uniform(0.08, 0.16),
            'lr': np.random.uniform(0.0001, 0.0003),
            'batch_size': int(np.random.choice([8, 10, 12])),
            'positive_class_weight': np.random.uniform(1.2, 1.8),
            'scheduler_factor': 0.25,
        }
    elif choice == 'high_lr':
        c = {
            'lookback_days': np.random.randint(19, 27),
            'hidden_dim': int(np.random.choice([128, 192, 224])),
            'dropout': np.random.uniform(0.10, 0.25),
            'lr': np.random.uniform(0.0007, 0.001),
            'batch_size': int(np.random.choice([24, 28, 32])),
            'positive_class_weight': np.random.uniform(1.2, 1.6),
            'scheduler_factor': 0.2,
        }
    elif choice == 'high_do':
        c = {
            'lookback_days': np.random.randint(20, 27),
            'hidden_dim': int(np.random.choice([96, 128, 160])),
            'dropout': np.random.uniform(0.20, 0.30),
            'lr': np.random.uniform(0.0003, 0.0007),
            'batch_size': int(np.random.choice([12, 16, 20])),
            'positive_class_weight': np.random.uniform(1.6, 2.0),
            'scheduler_factor': 0.3,
        }
    else:  # low_do
        c = {
            'lookback_days': np.random.randint(21, 28),
            'hidden_dim': int(np.random.choice([160, 192, 224, 256])),
            'dropout': np.random.uniform(0.05, 0.10),
            'lr': np.random.uniform(0.0004, 0.0006),
            'batch_size': int(np.random.choice([12, 16])),
            'positive_class_weight': np.random.uniform(1.0, 1.4),
            'scheduler_factor': 0.25,
        }
    configs.append(c)

# Normalize all values
for c in configs:
    c['lookback_days'] = int(c['lookback_days'])
    c['hidden_dim'] = int(c['hidden_dim'])
    c['dropout'] = round(float(c['dropout']), 2)
    c['lr'] = round(float(c['lr']), 6)
    c['batch_size'] = int(c['batch_size'])
    c['positive_class_weight'] = round(float(c['positive_class_weight']), 2)
    c['scheduler_factor'] = float(c['scheduler_factor'])

output_path = Path('scripts/smart_tuning_configs.json')
with open(output_path, 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'total_configs': len(configs),
        'strategy': 'Stratified: 60% HC, 30% explore, 10% edges',
        'configs': configs,
    }, f, indent=2)

print(f"Generated {len(configs)} configs -> {output_path}")

# Stats
lb = [c['lookback_days'] for c in configs]
hd = [c['hidden_dim'] for c in configs]
do = [c['dropout'] for c in configs]
lr = [c['lr'] for c in configs]
bs = [c['batch_size'] for c in configs]
pw = [c['positive_class_weight'] for c in configs]

print(f"Lookback: {min(lb)}-{max(lb)} (μ={np.mean(lb):.1f})")
print(f"Hidden:   {min(hd)}-{max(hd)} (μ={np.mean(hd):.1f})")
print(f"Dropout:  {min(do):.2f}-{max(do):.2f} (μ={np.mean(do):.2f})")
print(f"LR:       {min(lr):.6f}-{max(lr):.6f}")
print(f"BatchSz:  {min(bs)}-{max(bs)} (μ={np.mean(bs):.1f})")
print(f"PosWeight:{min(pw):.2f}-{max(pw):.2f} (μ={np.mean(pw):.2f})")
