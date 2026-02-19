#!/bin/bash

# Setup script for CIFAR-10 HPO project
# This script creates a conda environment with CUDA support

set -e  # Exit on error

echo "Setting up CIFAR-10 HPO conda environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create temporary environment file
TEMP_ENV_FILE=$(mktemp)
cp environment.yml "$TEMP_ENV_FILE"

# Check CUDA version (if available)
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c1-4)
    echo "Detected CUDA version: $CUDA_VERSION"
    
    # Adjust pytorch-cuda version based on CUDA version
    if command -v python3 &> /dev/null; then
        # Use Python to compare versions more reliably
        if python3 -c "import sys; exit(0 if float('$CUDA_VERSION') >= 12.0 else 1)"; then
            echo "Using pytorch-cuda=12.1"
            sed -i 's/pytorch-cuda=11.8/pytorch-cuda=12.1/' "$TEMP_ENV_FILE"
        else
            echo "Using pytorch-cuda=11.8"
        fi
    fi
else
    echo "CUDA not detected. Will install CPU-only PyTorch."
    sed -i '/pytorch-cuda/d' "$TEMP_ENV_FILE"
fi

# Create conda environment
echo "Creating conda environment from environment.yml..."
if conda env list | grep -q "^cifar10-hpo "; then
    echo "Environment 'cifar10-hpo' already exists. Removing it first..."
    conda env remove -n cifar10-hpo -y
fi

conda env create -f "$TEMP_ENV_FILE"

# Clean up temp file
rm "$TEMP_ENV_FILE"

# Activate environment and install additional requirements
echo "Installing additional requirements..."
eval "$(conda shell.bash hook)"
conda activate cifar10-hpo
pip install -r requirements.txt

echo ""
echo "✅ Setup complete! To activate the environment, run:"
echo "  conda activate cifar10-hpo"
echo ""
echo "To verify CUDA is available, run:"
echo "  python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \\\"N/A\\\"}')\""

