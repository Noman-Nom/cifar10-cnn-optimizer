# Setup Guide for CIFAR-10 HPO Project

## Quick Setup (Automated)

Run the setup script:
```bash
./setup_conda.sh
```

## Manual Setup

### Option 1: Using environment.yml (Recommended)

1. **Create conda environment with CUDA support:**
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the environment:**
   ```bash
   conda activate cifar10-hpo
   ```

3. **Install additional requirements:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Manual conda installation

1. **Create a new conda environment:**
   ```bash
   conda create -n cifar10-hpo python=3.10
   conda activate cifar10-hpo
   ```

2. **Install PyTorch with CUDA support:**
   ```bash
   # For CUDA 11.8
   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
   
   # For CUDA 12.1
   conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
   
   # For CPU only
   conda install pytorch torchvision -c pytorch
   ```

3. **Install other dependencies:**
   ```bash
   conda install numpy matplotlib seaborn scipy scikit-learn pandas -c conda-forge
   pip install tqdm
   ```

## Verify Installation

Check if CUDA is available:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

Check PyTorch version:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Running the Project

```bash
# Activate environment
conda activate cifar10-hpo

# Run training
python main.py
```

## Troubleshooting

### CUDA not detected
- Make sure you have NVIDIA drivers installed: `nvidia-smi`
- Check CUDA version compatibility with PyTorch
- Reinstall PyTorch with correct CUDA version

### Environment creation fails
- Update conda: `conda update conda`
- Try creating environment with `--force` flag: `conda env create -f environment.yml --force`

### Import errors
- Make sure environment is activated: `conda activate cifar10-hpo`
- Reinstall packages: `pip install -r requirements.txt --force-reinstall`

