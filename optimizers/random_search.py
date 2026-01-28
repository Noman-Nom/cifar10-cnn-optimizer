"""
Random Search hyperparameter optimization.
"""
import numpy as np
from .base_optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """
    Random Search optimizer for hyperparameter tuning.
    """
    
    def __init__(self, search_space, seed=42):
        """
        Initialize Random Search optimizer.
        
        Args:
            search_space (dict): Search space definition
            seed (int): Random seed
        """
        super().__init__(search_space, seed)
        self.name = "RandomSearch"
    
    def optimize(self, objective_function, n_iterations=25):
        """
        Run Random Search optimization.
        
        Args:
            objective_function: Function(config) -> score
            n_iterations (int): Number of random trials
        
        Returns:
            dict: Best configuration found
        """
        print(f"\n{'='*60}")
        print(f"Starting Random Search with {n_iterations} trials")
        print(f"{'='*60}\n")
        
        for i in range(n_iterations):
            # Sample random configuration
            config = self.sample_config()
            
            # Evaluate
            print(f"Trial {i+1}/{n_iterations}")
            print(f"Config: {config}")
            
            score = objective_function(config, trial_number=i)
            
            print(f"Validation Accuracy: {score:.4f}%\n")
            
            # Record
            self.record_evaluation(config, score, metadata={'trial': i+1})
        
        print(f"\n{'='*60}")
        print(f"Random Search Complete")
        print(f"Best Score: {self.best_score:.4f}%")
        print(f"Best Config: {self.best_config}")
        print(f"{'='*60}\n")
        
        return self.best_config

