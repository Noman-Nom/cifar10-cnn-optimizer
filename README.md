# CIFAR-10 Hyperparameter Optimization Framework

This project implements a comprehensive hyperparameter optimization framework comparing **Random Search** vs **Particle Swarm Optimization (PSO)** for training CNNs on the CIFAR-10 dataset.

## Overview

The framework implements a **two-phase exploration-exploitation approach**:
- **Phase 1 (Exploration)**: Fixed compute budget of 5,000 epochs per optimizer to find best hyperparameters
- **Phase 2 (Exploitation)**: Retrain best configurations for 15,000 epochs to achieve maximum performance

## Features

- ✅ Configurable CNN architecture with modern techniques (BatchNorm, Dropout)
- ✅ Two-phase optimization framework (exploration + exploitation)
- ✅ Random Search and PSO optimizers
- ✅ Proper train/val/test splits (45K/5K/10K)
- ✅ Data augmentation for CIFAR-10
- ✅ Early stopping with best model restoration
- ✅ Statistical analysis (Wilcoxon test, correlations)
- ✅ Publication-quality visualizations (300 DPI)
- ✅ Reproducible experiments with seed management
- ✅ GPU support with automatic device selection

## Project Structure

```
cifar10-hpo/
├── data/
│   ├── __init__.py
│   └── loaders.py              # CIFAR-10 data loading with augmentation
├── models/
│   ├── __init__.py
│   └── cnn.py                  # Configurable CNN architecture
├── optimizers/
│   ├── __init__.py
│   ├── base_optimizer.py       # Abstract base class
│   ├── random_search.py        # Random Search implementation
│   └── pso.py                  # Particle Swarm Optimization
├── utils/
│   ├── __init__.py
│   ├── trainer.py              # Training utilities with early stopping
│   └── visualization.py        # Plotting utilities
├── experiments/
│   ├── config.yaml             # Main configuration file
│   ├── run_experiment.py       # Main experiment orchestrator
│   ├── analyze_results.py      # Statistical analysis script
│   └── results/                # Experiment results (gitignored)
├── generate_publication_plots.py  # Generate publication figures
├── requirements.txt
├── environment.yml             # Conda environment
├── setup_conda.sh             # Setup script
└── README.md
```

## Setup

### Option 1: Automated Setup (Recommended)

```bash
./setup_conda.sh
conda activate cifar10-hpo
```

### Option 2: Manual Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate cifar10-hpo

# Install additional requirements
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check PyTorch version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## Configuration

Edit `experiments/config.yaml` to customize:

- **Number of runs**: `experiment.n_runs` (default: 3)
- **Device**: `experiment.device` (default: "cuda")
- **Search space**: Modify `search_space` section
- **Optimizer settings**: Adjust `optimizers.random_search` and `optimizers.pso`

### Search Space

The framework optimizes:
- `learning_rate`: [0.0001, 0.01] (log scale)
- `batch_size`: [32, 256] (log scale)
- `conv_channels_base`: [32, 128] (log scale)
- `num_conv_blocks`: [2, 4] (linear scale)
- `fc_hidden`: [64, 512] (linear scale)
- `dropout`: [0.1, 0.5] (linear scale)
- `weight_decay`: [0.00001, 0.001] (log scale)

## Running Experiments

### Run Complete Experiment

```bash
cd experiments
python run_experiment.py
```

This will:
1. Run Random Search (25 trials × 200 epochs = 5,000 epochs)
2. Run PSO (10 iterations × 10 particles × 50 epochs = 5,000 epochs)
3. Retrain best models for 15,000 epochs each
4. Save all results to `experiments/results/`

### Generate Visualizations

```bash
python generate_publication_plots.py <results_directory>
```

Example:
```bash
python generate_publication_plots.py experiments/results/CIFAR10_CNN_20240101_120000
```

### Statistical Analysis

```bash
cd experiments
python analyze_results.py <results_directory>
```

## Experimental Design

### Phase 1: Exploration (5,000 epochs each)

**Random Search:**
- 25 trials
- 200 epochs per trial
- Early stopping patience: 100 epochs
- Total: 25 × 200 = 5,000 epochs

**PSO:**
- 10 iterations
- 10 particles per iteration
- 50 epochs per particle evaluation
- Early stopping patience: 30 epochs
- Total: 10 × 10 × 50 = 5,000 epochs

### Phase 2: Exploitation (15,000 epochs)

- Retrain best configuration from Phase 1
- Early stopping patience: 50 epochs
- Learning rate scheduling (ReduceLROnPlateau)

## Results Structure

```
results/CIFAR10_CNN_YYYYMMDD_HHMMSS/
├── config.yaml
├── summary.json
├── publication_plots/
│   ├── fig1_comparison_boxplot.png
│   ├── fig2_convergence_curves.png
│   ├── fig3_training_curves.png
│   └── fig4_hyperparameter_importance.png
├── statistical_analysis/
│   ├── summary_statistics.csv
│   ├── pairwise_tests.csv
│   ├── hyperparameter_correlations.csv
│   └── efficiency_metrics.csv
├── random_search/
│   └── run_42/
│       ├── optimization_history.json
│       ├── training_history.json
│       └── best_model.pth
└── pso/
    └── run_42/
        ├── optimization_history.json
        ├── training_history.json
        └── best_model.pth
```

## Key Implementation Details

### Seed Management
- Each trial uses unique seed: `trial_seed = base_seed + trial_number`
- Ensures variation between trials while maintaining reproducibility

### GPU Utilization
- Automatic GPU detection and usage
- DataLoader with `pin_memory=True` for faster transfer
- Configurable `num_workers` for parallel data loading

### Early Stopping
- Monitors validation accuracy
- Restores best weights automatically
- Different patience for RS (100) vs PSO (30)

### Model Architecture
- Configurable number of convolutional blocks (2-4)
- Variable channel sizes
- Batch Normalization and Dropout
- Adaptive pooling for flexibility

## Expected Results

- **Accuracy Range**: 70-85% (CIFAR-10 is more challenging than MNIST)
- **Training Time**: ~2-4 hours per run on GPU (depends on hardware)
- **Variation**: Results should show variation between trials (not identical)

## Troubleshooting

### CUDA not detected
- Check NVIDIA drivers: `nvidia-smi`
- Verify CUDA compatibility with PyTorch
- Reinstall PyTorch with correct CUDA version

### Out of Memory
- Reduce `batch_size` in search space
- Reduce `num_workers` in config
- Use gradient accumulation if needed

### Identical Results
- Check seed management (each trial should have unique seed)
- Verify `evaluation_counter` is incrementing correctly

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{cifar10_hpo,
  title = {CIFAR-10 Hyperparameter Optimization Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/cifar10-hpo}
}
```

## License

[Specify your license here]

## Acknowledgments

- Based on previous work on MNIST hyperparameter optimization
- Uses PyTorch, torchvision, and other open-source libraries
