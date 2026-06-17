#!/bin/bash
# Stigmergic Latent Swarms - Model Training Script

# Prevent cd warnings
cd "$(dirname "$0")"

echo "========================================================="
echo "   ST-GVAE Model Training Setup"
echo "========================================================="

# 1. Check virtual environment
if [ ! -d "venv" ]; then
    echo "[!] Virtual environment 'venv' not found. Please run ./start.sh first to set up the dependencies."
    exit 1
fi

# 2. Check physical dataset
if [ ! -f "backend/data/spring_data.npz" ]; then
    echo "[*] Synthetic spring-mass dataset not found. Generating data..."
    venv/bin/python backend/data_generator.py
fi

# 3. Launch training
echo "[*] Launching training script..."
echo "Running: venv/bin/python backend/train.py $@"
echo ""

venv/bin/python backend/train.py "$@"

echo ""
echo "========================================================="
echo "Training completed."
echo "========================================================="
