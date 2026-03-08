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
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

def _load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)

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

def _make_pso_objective(search_space, max_epochs, es_patience, device):
    """Return objective closure for PSO particle evaluation."""
    def objective(config, trial_number=0):
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
    pso_cfg = cfg['optimizers']['pso']

    trial_max_epochs = 30
    trial_es_patience = 5
    retrain_max_epochs = training_cfg.get('retrain_max_epochs', 200)
    retrain_es_patience = training_cfg.get('early_stopping_patience', 30)
    population_size = pso_cfg.get('population_size', 10)
    n_iterations = pso_cfg.get('n_iterations', 3)
    w = pso_cfg.get('w', 0.7)
    c1 = pso_cfg.get('c1', 1.5)
    c2 = pso_cfg.get('c2', 1.5)

    # GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU Info: {gpu_info}")
    if gpu_info.get('available'):
        log_gpu_memory()

    # Baseline already done
    print("\nBaseline already complete — skipping.")

    # PSO is disabled guard
    if not pso_cfg.get('enabled', False):
        print("PSO is disabled in config. Exiting.")
        return

    pso_output_dir = os.path.join(output_dir, 'pso')
    os.makedirs(pso_output_dir, exist_ok=True)

    all_test_accs = []
    all_best_configs = []
    all_wall_times = []

    total_evals = population_size * n_iterations

    for seed in seeds:
        print("\n" + "=" * 60)
        print(f"PSO | seed={seed}")
        print("=" * 60)
        print(f"Starting PSO: {population_size} particles x {n_iterations} iterations = {total_evals} evaluations (seed={seed})")

        run_dir = os.path.join(pso_output_dir, f'run_seed{seed}')
        os.makedirs(run_dir, exist_ok=True)

        set_all_seeds(seed)
        t_start = time.time()

        # Fresh PSO instance per seed
        pso = ParticleSwarmOptimizer(
            search_space=search_space,
            population_size=population_size,
            w=w,
            c1=c1,
            c2=c2,
            seed=seed
        )

        objective_fn = _make_pso_objective(search_space, trial_max_epochs, trial_es_patience, device)

        opt_history = []
        param_names = list(search_space.keys())
        evaluation_counter = 0

        for iteration in range(n_iterations):
            print(f"\nPSO Iteration {iteration + 1}/{n_iterations}")
            print("-" * 60)

            for particle_idx in range(pso.population_size):
                config = pso._vector_to_config(pso.particles[particle_idx], param_names)
                print(f"  Particle {particle_idx + 1}/{pso.population_size} | Config: {config}")

                try:
                    score = objective_fn(config, trial_number=evaluation_counter)
                except Exception as exc:
                    print(f"  [Particle {particle_idx + 1}] ERROR: {exc}")
                    score = 0.0

                evaluation_counter += 1
                print(f"  Val Accuracy: {score:.4f}%")

                # Update personal best
                if score > pso.personal_best_scores[particle_idx]:
                    pso.personal_best_scores[particle_idx] = score
                    pso.personal_best_positions[particle_idx] = pso.particles[particle_idx].copy()

                # Update global best
                if score > pso.global_best_score:
                    pso.global_best_score = score
                    pso.global_best_position = pso.particles[particle_idx].copy()
                    pso.best_config = config.copy()
                    pso.best_score = score
                    print(f"  *** New global best: {score:.4f}% ***")

                pso.record_evaluation(config, score, metadata={
                    'iteration': iteration + 1,
                    'particle': particle_idx + 1
                })
                opt_history.append({
                    'iteration': iteration + 1,
                    'particle': particle_idx + 1,
                    'config': _to_python(config),
                    'score': float(score)
                })

            # Update velocities and positions
            for particle_idx in range(pso.population_size):
                r1 = np.random.random(len(param_names))
                r2 = np.random.random(len(param_names))

                cognitive = pso.c1 * r1 * (pso.personal_best_positions[particle_idx] - pso.particles[particle_idx])
                social = pso.c2 * r2 * (pso.global_best_position - pso.particles[particle_idx])
                pso.velocities[particle_idx] = (
                    pso.w * pso.velocities[particle_idx] + cognitive + social
                )

                # Clamp velocity
                pso.velocities[particle_idx] = np.clip(pso.velocities[particle_idx], -pso.v_max, pso.v_max)

                # Update position
                pso.particles[particle_idx] += pso.velocities[particle_idx]

                # Reflect at boundaries
                for d in range(len(pso.particles[particle_idx])):
                    if pso.particles[particle_idx][d] < 0.0:
                        pso.particles[particle_idx][d] = abs(pso.particles[particle_idx][d])
                        pso.velocities[particle_idx][d] *= pso.boundary_damping
                    elif pso.particles[particle_idx][d] > 1.0:
                        pso.particles[particle_idx][d] = 2.0 - pso.particles[particle_idx][d]
                        pso.velocities[particle_idx][d] *= pso.boundary_damping
                    pso.particles[particle_idx][d] = np.clip(pso.particles[particle_idx][d], 0.0, 1.0)

            pso.swarm_convergence.append(float(pso.global_best_score))
            print(f"\n  Iteration {iteration + 1} complete | Global Best: {pso.global_best_score:.4f}%")
            print(f"  Best Config so far: {pso.best_config}")

        print(f"\n{'=' * 60}")
        print(f"PSO Complete | Best Score: {pso.best_score:.4f}%")
        print(f"Best Config: {pso.best_config}")
        print(f"{'=' * 60}\n")

        # Save optimization history
        with open(os.path.join(run_dir, 'optimization_history.json'), 'w') as f:
            json.dump(opt_history, f, indent=2)

        # Save best config
        with open(os.path.join(run_dir, 'best_config.json'), 'w') as f:
            json.dump(_to_python(pso.best_config), f, indent=2)

        # Save swarm convergence
        with open(os.path.join(run_dir, 'swarm_convergence.json'), 'w') as f:
            json.dump({'seed': seed, 'global_best_per_iteration': pso.swarm_convergence}, f, indent=2)

        # Retrain best config
        print(f"\nRetraining best config for seed={seed}...")
        test_acc, train_history = _retrain_best(
            best_config=pso.best_config,
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
        all_best_configs.append(_to_python(pso.best_config))
        all_wall_times.append(wall_time)

    # -----------------------------------------------------------------------
    # PSO summary
    # -----------------------------------------------------------------------
    pso_mean = float(np.mean(all_test_accs))
    pso_std = float(np.std(all_test_accs, ddof=1))

    pso_summary = {
        'test_accuracies': all_test_accs,
        'mean': pso_mean,
        'std': pso_std,
        'seeds': seeds,
        'best_configs': all_best_configs,
        'wall_times_seconds': all_wall_times
    }

    with open(os.path.join(pso_output_dir, 'pso_summary.json'), 'w') as f:
        json.dump(pso_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("PSO COMPLETE")
    print(f"Mean test accuracy: {pso_mean:.4f}% +/- {pso_std:.4f}%")
    print(f"All test accuracies: {all_test_accs}")
    print("=" * 60)


if __name__ == '__main__':
    main()