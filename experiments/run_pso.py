"""
Run PSO (Particle Swarm Optimization) for CIFAR-10 CNN hyperparameter optimization (Q2).

Entry point:
    python experiments/run_pso.py

Outputs are written to q2_results/pso/
Baseline is skipped — already completed by run_random_search.py
"""
import sys
import os
import json
import time

import yaml
import numpy as np
import torch
import torch.optim as optim

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.cnn import create_cnn_from_config
from optimizers.pso import ParticleSwarmOptimizer
from utils.trainer import ModelTrainer, EarlyStopping
from utils.seed_utils import set_all_seeds
from utils.gpu_utils import get_gpu_info, log_gpu_memory
from data.loaders import get_cifar10_loaders


# ---------------------------------------------------------------------------
# Helpers  (identical to run_random_search.py)
# ---------------------------------------------------------------------------

def _to_python(obj):
    """Recursively convert numpy scalars to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,...
    print(f"Mean test accuracy: {pso_mean:.4f}% ± {pso_std:.4f}%")
    print(f"All test accuracies: {all_test_accs}")
    print("=" * 60)


if __name__ == '__main__':
    main()