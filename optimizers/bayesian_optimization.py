"""
Bayesian Optimization hyperparameter tuning using Optuna.
"""
import optuna
import numpy as np
from .base_optimizer import BaseOptimizer


class BayesianOptimization(BaseOptimizer):
    """
    Bayesian Optimization using Optuna for hyperparameter tuning.
    """
    
    def __init__(self, search_space, seed=42):
        """
        Initialize Bayesian Optimization.
        
        Args:
            search_space (dict): Search space definition
            seed (int): Random seed
        """
        super().__init__(search_space, seed)
        self.name = "BayesianOptimization"
        
        # Use TPE Sampler with the prescribed seed
        self.sampler = optuna.samplers.TPESampler(seed=seed)
        # Suppress Optuna's default info logging for cleaner console output
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def optimize(self, objective_function, n_iterations=25):
        """
        Run Bayesian Optimization.
        
        Args:
            objective_function: Function(config) -> score
            n_iterations (int): Number of trials
        
        Returns:
            dict: Best configuration found
        """
        print(f"\n{'='*60}")
        print(f"Starting Bayesian Optimization with {n_iterations} trials")
        print(f"{'='*60}\n")
        
        # Create an Optuna study to maximize the validation accuracy
        study = optuna.create_study(direction="maximize", sampler=self.sampler)
        
        def optuna_objective(trial):
            config = {}
            for param_name, param_config in self.search_space.items():
                param_type = param_config.get('type')
                
                if param_type == 'int':
                    low = param_config['min']
                    high = param_config['max']
                    log = param_config.get('scale', 'linear') == 'log'
                    config[param_name] = trial.suggest_int(param_name, low, high, log=log)
                    
                elif param_type == 'float':
                    low = param_config['min']
                    high = param_config['max']
                    log = param_config.get('scale', 'linear') == 'log'
                    config[param_name] = trial.suggest_float(param_name, low, high, log=log)
                    
                elif param_type == 'categorical':
                    choices = param_config['choices']
                    config[param_name] = trial.suggest_categorical(param_name, choices)
            
            trial_num = trial.number
            print(f"Trial {trial_num+1}/{n_iterations}")
            print(f"Config: {config}")
            
            # Evaluate using the external objective function
            score = objective_function(config, trial_number=trial_num)
            
            print(f"Validation Accuracy: {score:.4f}%\n")
            
            # Record it manually to match the standard history format of other optimizers
            self.record_evaluation(config, score, metadata={'trial': trial_num+1})
            
            return score

        # Execute optimization
        study.optimize(optuna_objective, n_trials=n_iterations)
        
        print(f"\n{'='*60}")
        print(f"Bayesian Optimization Complete")
        print(f"Best Score: {self.best_score:.4f}%")
        print(f"Best Config: {self.best_config}")
        print(f"{'='*60}\n")
        
        return self.best_config
