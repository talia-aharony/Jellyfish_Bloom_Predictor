#!/bin/bash
# Simple structure verification

echo "================================================================================"
echo "JELLYFISH BLOOM PREDICTOR - SYSTEM CHECK"
echo "================================================================================"
echo ""

# Test 1: Check core modules
echo "TEST 1: Checking core modules..."
modules=("train.py" "predictor.py" "models.py" "data_loader.py" "data_loader_forecasting.py")
for module in "${modules[@]}"; do
    if [ -f "$module" ]; then
        echo "  ✓ $module exists"
    else
        echo "  ✗ $module NOT FOUND"
    fi
done
echo ""

# Test 2: Check data
echo "TEST 2: Checking data files..."
if [ -d "data" ]; then
    echo "  ✓ data/ directory exists"
    find data -type f | wc -l | xargs echo "    Files found:"
else
    echo "  ! data/ directory not found"
fi
echo ""

# Test 3: Check scripts
echo "TEST 3: Checking scripts..."
if [ -d "scripts" ]; then
    echo "  ✓ scripts/ directory exists"
    ls scripts/*.py | xargs -I {} basename {} | sed 's/^/    - /'
else
    echo "  ✗ scripts/ directory not found"
fi
echo ""

# Test 4: Check jellyfish package
echo "TEST 4: Checking jellyfish package..."
if [ -d "jellyfish" ]; then
    echo "  ✓ jellyfish/ directory exists"
    ls jellyfish/*.py | xargs -I {} basename {} | sed 's/^/    - /'
else
    echo "  ✗ jellyfish/ directory not found"
fi
echo ""

# Test 5: Check venv
echo "TEST 5: Checking virtual environment..."
if [ -d ".venv" ]; then
    echo "  ✓ .venv/ directory exists"
    python_path=".venv/bin/python"
    if [ -f "$python_path" ]; then
        echo "  ✓ Python executable found"
    fi
else
    echo "  ! .venv/ directory not found"
fi
echo ""

echo "================================================================================"
echo "SYSTEM CHECK COMPLETE"
echo "================================================================================"
echo ""
echo "Project structure is ready. To run:"
echo "  source .venv/bin/activate"
echo "  python scripts/train.py     # Train models"
echo "  python scripts/predict.py   # Make predictions"
echo "  python scripts/evaluate.py  # Evaluate models"
echo ""
