"""
Run Random Search for CIFAR-10 CNN hyperparameter optimization (Q2).

Entry point:
    python experiments/run_random_search.py

Outputs are written to the directory specified in experiments/config.yaml
(default: q2_results/).
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
from optimizers.random_search import RandomSearch
from utils.trainer import ModelTrainer, EarlyStopping
from utils.seed_utils import set_all_seeds
from utils.gpu_utils import get_gpu_info, log_gpu_memory
from data.loaders import get_cifar10_loaders
from experiments.baseline_runner import run_baseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_python(obj):
    """Recursively convert numpy scalars to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)


def _make_objective(search_space, max_epochs, es_patience, device):
    """
    Return a closure that trains a model with the given config and returns
    validation accuracy.  Each trial uses a unique numpy seed derived from
    the seed stored in the closure plus the trial number, so every call
    produces a genuinely different configuration.
    """
    def objective(config, trial_number=0):
        # Fix: re-seed numpy per trial to avoid duplicate configs
        np.random.seed(int(time.time() * 1000) % (2 ** 31) + trial_number * 1000)

        try:
            batch_size = int(config['batch_size'])
            train_loader, val_loader, _ = get_cifar10_loaders(batch_size=batch_size)

            model = create_cnn_from_config(config)
            optimizer = optim.Adam(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )

            trainer = ModelTrainer(model=model, device=device, optimizer=optimizer, verbose=False)
            early_stopping = EarlyStopping(patience=es_patience, mode='max',
                                           restore_best_weights=True)

            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                max_epochs=max_epochs,
                early_stopping=early_stopping
            )

            _, val_acc = trainer.validate(val_loader)
            return float(val_acc)

        except Exception as exc:
            print(f"  [Trial {trial_number}] ERROR: {exc}")
            return 0.0

    return objective


def _retrain_best(best_config, seed, max_epochs, es_patience, device):
    """Retrain best config from scratch and return (test_acc, training_history)."""
    set_all_seeds(seed)
    batch_size = int(best_config['batch_size'])
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    model = create_cnn_from_config(best_config)
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_config['learning_rate'],
        weight_decay=best_config['weight_decay']
    )

    trainer = ModelTrainer(model=model, device=device, optimizer=optimizer, verbose=True)
    early_stopping = EarlyStopping(patience=es_patience, mode='max',
                                   restore_best_weights=True)

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=max_epochs,
        early_stopping=early_stopping
    )

    _, test_acc = trainer.evaluate(test_loader)
    return float(test_acc), history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = _load_config()

    exp = cfg['experiment']
    seeds = exp['seeds']
    output_dir = exp['output_dir']
    device_str = exp.get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    search_space = cfg['search_space']
    training_cfg = cfg['training']
    rs_cfg = cfg['optimizers']['random_search']
    baseline_cfg = cfg['baseline']

    trial_max_epochs = 30
    trial_es_patience = 5
    retrain_max_epochs = training_cfg.get('retrain_max_epochs', 200)
    retrain_es_patience = training_cfg.get('early_stopping_patience', 30)
    n_iterations = rs_cfg.get('n_iterations', 15)

    # GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU Info: {gpu_info}")
    if gpu_info.get('available'):
        log_gpu_memory()

    # -----------------------------------------------------------------------
    # Baseline
    # -----------------------------------------------------------------------
    if baseline_cfg.get('enabled', True):
        print("\n" + "=" * 60)
        print("RUNNING BASELINE")
        print("=" * 60)
        baseline_output_dir = os.path.join(output_dir, 'baseline')
        run_baseline(
            baseline_config=baseline_cfg['config'],
            seeds=seeds,
            max_epochs=retrain_max_epochs,
            early_stopping_patience=retrain_es_patience,
            output_dir=baseline_output_dir,
            device=device
        )

    # -----------------------------------------------------------------------
    # Random Search
    # -----------------------------------------------------------------------
    if not rs_cfg.get('enabled', True):
        print("Random Search is disabled in config. Exiting.")
        return

    rs_output_dir = os.path.join(output_dir, 'random_search')
    os.makedirs(rs_output_dir, exist_ok=True)

    all_test_accs = []
    all_best_configs = []
    all_wall_times = []

    for seed in seeds:
        print("\n" + "=" * 60)
        print(f"RANDOM SEARCH | seed={seed}")
        print("=" * 60)

        run_dir = os.path.join(rs_output_dir, f'run_seed{seed}')
        os.makedirs(run_dir, exist_ok=True)

        set_all_seeds(seed)
        t_start = time.time()

        # --- Search phase ---
        rs = RandomSearch(search_space=search_space, seed=seed)

        # Patch: override sample_config per trial to avoid duplicate configs
        original_optimize = rs.optimize

        opt_history = []

        def patched_objective(config, trial_number=0):
            # Each trial uses seed + trial_number * 1000 for diversity
            np.random.seed(seed + trial_number * 1000)
            return _make_objective(search_space, trial_max_epochs,
                                   trial_es_patience, device)(config, trial_number)

        def patched_optimize(objective_fn, n_iter):
            print(f"\n{'='*60}")
            print(f"Starting Random Search with {n_iter} trials (seed={seed})")
            print(f"{'='*60}\n")
            for i in range(n_iter):
                np.random.seed(seed + i * 1000)
                config = rs.sample_config()
                print(f"Trial {i+1}/{n_iter} | Config: {config}")
                try:
                    score = objective_fn(config, trial_number=i)
                except Exception as exc:
                    print(f"  [Trial {i+1}] ERROR: {exc}")
                    score = 0.0
                print(f"  Val Accuracy: {score:.4f}%")
                rs.record_evaluation(config, score, metadata={'trial': i + 1})
                opt_history.append({'config': _to_python(config), 'score': float(score),
                                    'trial': i + 1})
                if score == rs.best_score:
                    print(f"  *** New best: {score:.4f}% ***")

            print(f"\n{'='*60}")
            print(f"RS Complete | Best Score: {rs.best_score:.4f}%")
            print(f"Best Config: {rs.best_config}")
            print(f"{'='*60}\n")
            return rs.best_config

        best_config = patched_optimize(patched_objective, n_iterations)

        # Save optimization history
        with open(os.path.join(run_dir, 'optimization_history.json'), 'w') as f:
            json.dump(opt_history, f, indent=2)

        # Save best config
        with open(os.path.join(run_dir, 'best_config.json'), 'w') as f:
            json.dump(_to_python(best_config), f, indent=2)

        # --- Retrain phase ---
        print(f"\nRetraining best config for seed={seed}...")
        test_acc, train_history = _retrain_best(
            best_config=best_config,
            seed=seed,
            max_epochs=retrain_max_epochs,
            es_patience=retrain_es_patience,
            device=device
        )
        wall_time = time.time() - t_start

        print(f"Seed={seed} test accuracy: {test_acc:.4f}% | Wall time: {wall_time:.1f}s")
        if gpu_info.get('available'):
            log_gpu_memory()

        # Save per-run outputs
        with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
            json.dump(_to_python(train_history), f, indent=2)

        with open(os.path.join(run_dir, 'test_accuracy.json'), 'w') as f:
            json.dump({'seed': seed, 'test_accuracy': test_acc}, f, indent=2)

        with open(os.path.join(run_dir, 'wall_time_seconds.json'), 'w') as f:
            json.dump({'seed': seed, 'wall_time_seconds': wall_time}, f, indent=2)

        all_test_accs.append(test_acc)
        all_best_configs.append(_to_python(best_config))
        all_wall_times.append(wall_time)

    # -----------------------------------------------------------------------
    # RS summary
    # -----------------------------------------------------------------------
    rs_mean = float(np.mean(all_test_accs))
    rs_std = float(np.std(all_test_accs, ddof=1))

    rs_summary = {
        'test_accuracies': all_test_accs,
        'mean': rs_mean,
        'std': rs_std,
        'seeds': seeds,
        'best_configs': all_best_configs,
        'wall_times_seconds': all_wall_times
    }

    with open(os.path.join(rs_output_dir, 'rs_summary.json'), 'w') as f:
        json.dump(rs_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("RANDOM SEARCH COMPLETE")
    print(f"Mean test accuracy: {rs_mean:.4f}% ± {rs_std:.4f}%")
    print(f"All test accuracies: {all_test_accs}")
    print("=" * 60)


if __name__ == '__main__':
    main()
