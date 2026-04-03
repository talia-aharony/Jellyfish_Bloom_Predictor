#!/usr/bin/env python3
"""
Simple test to verify the program structure is correct.
"""

import sys
import os

print("=" * 80)
print("JELLYFISH BLOOM PREDICTOR - SYSTEM CHECK")
print("=" * 80)
print()

# Test 1: Check imports
print("TEST 1: Checking required packages...")
required_packages = ['numpy', 'pandas', 'torch', 'sklearn', 'matplotlib']
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg} available")
    except ImportError:
        print(f"  ✗ {pkg} NOT available")
print()

# Test 2: Check data exists
print("TEST 2: Checking data files...")
data_dir = "data"
if os.path.exists(data_dir):
    print(f"  ✓ {data_dir}/ directory exists")
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"    Found {len(subdirs)} subdirectories:")
    for subdir in subdirs:
        print(f"      - {subdir}")
else:
    print(f"  ! Warning: {data_dir}/ directory not found")
print()

# Test 3: Check core modules
print("TEST 3: Checking core modules...")
modules_to_check = [
    "train.py",
    "predictor.py",
    "models.py",
    "data_loader.py"
]
for module in modules_to_check:
    if os.path.exists(module):
        print(f"  ✓ {module} exists")
    else:
        print(f"  ✗ {module} NOT FOUND")
print()

# Test 4: Check scripts
print("TEST 4: Checking scripts...")
script_dir = "scripts"
if os.path.exists(script_dir):
    print(f"  ✓ {script_dir}/ directory exists")
    scripts = [f for f in os.listdir(script_dir) if f.endswith(".py")]
    print(f"    Found {len(scripts)} scripts:")
    for script in scripts:
        print(f"      - {script}")
else:
    print(f"  ✗ {script_dir}/ directory not found")
print()

# Test 5: Check jellyfish package
print("TEST 5: Checking jellyfish package...")
jellyfish_dir = "jellyfish"
if os.path.exists(jellyfish_dir):
    print(f"  ✓ {jellyfish_dir}/ directory exists")
    modules = [f for f in os.listdir(jellyfish_dir) if f.endswith(".py")]
    print(f"    Found {len(modules)} modules:")
    for module in modules:
        print(f"      - {module}")
else:
    print(f"  ✗ {jellyfish_dir}/ directory not found")
print()

print("=" * 80)
print("SYSTEM CHECK COMPLETE")
print("=" * 80)
print()
print("To train models:")
print("  python scripts/train.py")
print()
print("To make predictions:")
print("  python scripts/predict.py")
print()
print("To evaluate models:")
print("  python scripts/evaluate.py")
print()
