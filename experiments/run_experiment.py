"""
Main experiment runner for CIFAR-10 hyperparameter optimization.
Implements two-phase exploration-exploitation framework.
"""
import os
import sys
import json
import yaml
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn

# Add parent directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.loaders import get_cifar10_loaders
from models.cnn import create_cnn_from_config
from utils.trainer import ModelTrainer, EarlyStopping
from optimizers.random_search import RandomSearch
from optimizers.pso import ParticleSwarmOptimizer
from optimizers.bayesian_optimization import BayesianOptimization


class HyperparameterExperiment:
    """
    Main experiment orchestrator for hyperparameter optimization.
    """
    
    def __init__(self, config_path=None):
        """Initialize experiment from config file."""
        # Default to config.yaml in the same directory as this script
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'
        else:
            config_path = Path(config_path)
            # If relative path, try relative to script first, then project root
            if not config_path.is_absolute():
                script_dir_config = Path(__file__).parent / config_path
                if script_dir_config.exists():
                    config_path = script_dir_config
                else:
                    # Try from project root
                    project_root_config = Path(__file__).parent.parent / config_path
                    if project_root_config.exists():
                        config_path = project_root_config
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        device_name = self.config['experiment']['device']
        if device_name == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Create results directory
        self.results_dir = Path(self.config['experiment']['results_dir'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.results_dir / f"CIFAR10_CNN_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.run_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        
        # Evaluation counter for unique seeds
        self.evaluation_counter = 0
    
    def set_seed(self, seed):
        """Set all random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def objective_function(self, config, optimizer_name='random_search', trial_number=0):
        """
        Objective function for hyperparameter evaluation.
        
        CRITICAL: Uses unique seed per trial: trial_seed = base_seed + trial_number
        
        Args:
            config: Hyperparameter configuration dict
            optimizer_name: Name of optimizer ('random_search' or 'pso')
            trial_number: Trial number for unique seed
        
        Returns:
            float: Validation accuracy
        """
        base_seed = self.config['experiment']['base_seed']
        trial_seed = base_seed + trial_number
        
        # Set unique seed for this trial
        self.set_seed(trial_seed)
        
        # Get training parameters based on optimizer
        if optimizer_name == 'random_search':
            max_epochs = self.config['training']['max_epochs']
            patience = self.config['training']['early_stopping_patience']
        elif optimizer_name == 'bo':
            max_epochs = self.config['bo_training']['max_epochs']
            patience = self.config['bo_training']['early_stopping_patience']
        else:  # PSO
            max_epochs = self.config['pso_training']['max_epochs']
            patience = self.config['pso_training']['early_stopping_patience']
        
        # Create data loaders
        batch_size = config['batch_size']
        train_loader, val_loader, _ = get_cifar10_loaders(
            batch_size=batch_size,
            val_split=self.config['dataset']['val_split'],
            num_workers=self.config['experiment']['num_workers'],
            pin_memory=(self.device.type == 'cuda'),
            data_dir=self.config['dataset']['data_dir'],
            seed=trial_seed
        )
        
        # Create model
        model = create_cnn_from_config(config)
        
        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Add learning rate scheduler for faster convergence
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            device=self.device,
            optimizer=optimizer,
            scheduler=scheduler,
            verbose=False  # Reduce output during optimization
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=patience,
            mode='max',
            restore_best_weights=True
        )
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            early_stopping=early_stopping
        )
        
        # Get best validation accuracy
        val_acc = max(history['val_acc'])
        
        return val_acc
    
    def run_optimizer(self, optimizer_name, base_seed):
        """
        Run a single optimizer (Random Search or PSO).
        
        Args:
            optimizer_name: 'random_search' or 'pso'
            base_seed: Base seed for this run
        
        Returns:
            dict: Results including best config and history
        """
        print(f"\n{'='*80}")
        print(f"Running {optimizer_name.upper()} with base_seed={base_seed}")
        print(f"{'='*80}\n")
        
        self.set_seed(base_seed)
        self.evaluation_counter = 0
        
        # Create optimizer
        search_space = self.config['search_space']
        
        if optimizer_name == 'random_search':
            opt_config = self.config['optimizers']['random_search']
            optimizer = RandomSearch(
                search_space=search_space,
                seed=base_seed
            )
            n_iterations = opt_config['n_iterations']
        elif optimizer_name == 'bo':
            opt_config = self.config['optimizers']['bo']
            optimizer = BayesianOptimization(
                search_space=search_space,
                seed=base_seed
            )
            n_iterations = opt_config['n_iterations']
        else:  # PSO
            opt_config = self.config['optimizers']['pso']
            optimizer = ParticleSwarmOptimizer(
                search_space=search_space,
                population_size=opt_config['population_size'],
                w=opt_config['w'],
                c1=opt_config['c1'],
                c2=opt_config['c2'],
                seed=base_seed
            )
            n_iterations = opt_config['n_iterations']
        
        # Create objective function wrapper
        def objective_wrapper(config, trial_number=None, **kwargs):
            # Use provided trial_number if available, otherwise use evaluation_counter
            if trial_number is not None:
                trial_num = trial_number
            else:
                trial_num = self.evaluation_counter
                self.evaluation_counter += 1
            
            score = self.objective_function(
                config, 
                optimizer_name=optimizer_name,
                trial_number=trial_num
            )
            return score
        
        # Run optimization
        best_config = optimizer.optimize(objective_wrapper, n_iterations=n_iterations)
        
        # Return results
        return {
            'optimizer': optimizer_name,
            'base_seed': base_seed,
            'best_config': best_config,
            'best_score': optimizer.best_score,
            'history': optimizer.history
        }
    
    def retrain_best_model(self, best_config, optimizer_name, base_seed, run_dir):
        """
        Retrain best model from Phase 1 (exploitation phase).
        
        Args:
            best_config: Best hyperparameter configuration
            optimizer_name: Name of optimizer
            base_seed: Base seed
            run_dir: Directory to save results
        
        Returns:
            dict: Retraining results
        """
        print(f"\n{'='*80}")
        print(f"Retraining best {optimizer_name.upper()} model (Exploitation Phase)")
        print(f"{'='*80}\n")
        
        retrain_seed = base_seed + 10000  # Different seed for retraining
        self.set_seed(retrain_seed)
        
        # Create data loaders
        batch_size = best_config['batch_size']
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            batch_size=batch_size,
            val_split=self.config['dataset']['val_split'],
            num_workers=self.config['experiment']['num_workers'],
            pin_memory=(self.device.type == 'cuda'),
            data_dir=self.config['dataset']['data_dir'],
            seed=retrain_seed
        )
        
        # Create model
        model = create_cnn_from_config(best_config)
        
        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=best_config['learning_rate'],
            weight_decay=best_config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            device=self.device,
            optimizer=optimizer,
            scheduler=scheduler,
            verbose=True
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config['retraining']['early_stopping_patience'],
            mode='max',
            restore_best_weights=True
        )
        
        # Save path
        save_path = run_dir / 'best_model.pth' if self.config['experiment']['save_models'] else None
        
        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=self.config['retraining']['max_epochs'],
            early_stopping=early_stopping,
            save_path=save_path
        )
        
        # Evaluate on test set
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        print(f"\nFinal Test Accuracy: {test_acc:.4f}%")
        
        # Save training history
        with open(run_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        return {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'training_history': history,
            'best_config': best_config
        }
    
    def run(self):
        """Run complete experiment with multiple runs."""
        n_runs = self.config['experiment']['n_runs']
        base_seeds = [self.config['experiment']['base_seed'] + i for i in range(n_runs)]
        
        all_results = {
            'random_search': [],
            'bo': [],
            'pso': []
        }
        
        # Run each optimizer for each seed
        for seed in base_seeds:
            # Random Search
            if self.config['optimizers']['random_search']['enabled']:
                rs_dir = self.run_dir / 'random_search' / f'run_{seed}'
                rs_dir.mkdir(parents=True, exist_ok=True)
                
                rs_results = self.run_optimizer('random_search', seed)
                
                # Save optimization history
                with open(rs_dir / 'optimization_history.json', 'w') as f:
                    json.dump(rs_results['history'], f, indent=2)
                
                # Retrain best model
                retrain_results = self.retrain_best_model(
                    rs_results['best_config'],
                    'random_search',
                    seed,
                    rs_dir
                )
                
                rs_results['retrain_results'] = retrain_results
                all_results['random_search'].append(rs_results)
            
            # BO
            if self.config['optimizers'].get('bo', {}).get('enabled', False):
                bo_dir = self.run_dir / 'bo' / f'run_{seed}'
                bo_dir.mkdir(parents=True, exist_ok=True)
                
                bo_results = self.run_optimizer('bo', seed)
                
                # Save optimization history
                with open(bo_dir / 'optimization_history.json', 'w') as f:
                    json.dump(bo_results['history'], f, indent=2)
                
                # Retrain best model
                retrain_results = self.retrain_best_model(
                    bo_results['best_config'],
                    'bo',
                    seed,
                    bo_dir
                )
                
                bo_results['retrain_results'] = retrain_results
                all_results['bo'].append(bo_results)
            
            # PSO
            if self.config['optimizers']['pso']['enabled']:
                pso_dir = self.run_dir / 'pso' / f'run_{seed}'
                pso_dir.mkdir(parents=True, exist_ok=True)
                
                pso_results = self.run_optimizer('pso', seed)
                
                # Save optimization history
                with open(pso_dir / 'optimization_history.json', 'w') as f:
                    json.dump(pso_results['history'], f, indent=2)
                
                # Retrain best model
                retrain_results = self.retrain_best_model(
                    pso_results['best_config'],
                    'pso',
                    seed,
                    pso_dir
                )
                
                pso_results['retrain_results'] = retrain_results
                all_results['pso'].append(pso_results)
        
        # Save summary
        summary = {
            'random_search': {
                'test_accuracies': [r['retrain_results']['test_accuracy'] for r in all_results['random_search']],
                'best_configs': [r['best_config'] for r in all_results['random_search']]
            },
            'bo': {
                'test_accuracies': [r['retrain_results']['test_accuracy'] for r in all_results['bo'] if 'retrain_results' in r],
                'best_configs': [r['best_config'] for r in all_results['bo']]
            },
            'pso': {
                'test_accuracies': [r['retrain_results']['test_accuracy'] for r in all_results['pso']],
                'best_configs': [r['best_config'] for r in all_results['pso']]
            }
        }
        
        with open(self.run_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print("Experiment Complete!")
        print(f"Results saved to: {self.run_dir}")
        print(f"{'='*80}\n")
        
        return all_results


if __name__ == "__main__":
    experiment = HyperparameterExperiment()
    results = experiment.run()

