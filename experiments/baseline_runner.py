"""
Baseline runner: trains the default config over multiple seeds and records test accuracies.
"""
import sys
import os
import json

import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.cnn import create_cnn_from_config
from utils.trainer import ModelTrainer, EarlyStopping
from utils.seed_utils import set_all_seeds
from data.loaders import get_cifar10_loaders


def run_baseline(baseline_config, seeds, max_epochs, early_stopping_patience,
                 output_dir, device):
    """
    Train the default baseline config for each seed and collect test accuracies.

    Args:
        baseline_config (dict): Default hyperparameter configuration.
        seeds (list): List of integer seeds to use.
        max_epochs (int): Max training epochs per run.
        early_stopping_patience (int): Early stopping patience.
        output_dir (str): Directory to save baseline outputs.
        device (torch.device): Device to train on.

    Returns:
        list: Test accuracies for each seed (floats).
    """
    os.makedirs(output_dir, exist_ok=True)
    test_accuracies = []

    for seed in seeds:
        print(f"\n--- Baseline | seed={seed} ---")
        set_all_seeds(seed)

        train_loader, val_loader, test_loader = get_cifar10_loaders(
            batch_size=baseline_config['batch_size']
        )

        model = create_cnn_from_config(baseline_config)

        optimizer = optim.Adam(
            model.parameters(),
            lr=baseline_config['learning_rate'],
            weight_decay=baseline_config['weight_decay']
        )

        trainer = ModelTrainer(model=model, device=device, optimizer=optimizer, verbose=False)
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max',
                                       restore_best_weights=True)

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            early_stopping=early_stopping
        )

        _, test_acc = trainer.evaluate(test_loader)
        test_acc = float(test_acc)
        test_accuracies.append(test_acc)
        print(f"Baseline seed={seed} test accuracy: {test_acc:.4f}%")

    baseline_output = {
        'test_accuracies': test_accuracies,
        'mean': float(sum(test_accuracies) / len(test_accuracies)),
        'seeds': seeds,
        'config': baseline_config
    }

    with open(os.path.join(output_dir, 'test_accuracies.json'), 'w') as f:
        json.dump(test_accuracies, f, indent=2)

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(baseline_output, f, indent=2)

    print(f"\nBaseline mean test accuracy: {baseline_output['mean']:.4f}%")
    return test_accuracies
