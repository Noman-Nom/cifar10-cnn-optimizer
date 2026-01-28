"""
Base class for hyperparameter optimizers.
"""
import numpy as np
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Abstract base class for hyperparameter optimization algorithms.
    """
    
    def __init__(self, search_space, seed=42):
        """
        Initialize optimizer.
        
        Args:
            search_space (dict): Dictionary defining search space for each hyperparameter
            seed (int): Random seed
        """
        self.search_space = search_space
        self.seed = seed
        np.random.seed(seed)
        
        self.history = []
        self.best_config = None
        self.best_score = -np.inf
    
    def sample_parameter(self, param_name, param_config):
        """
        Sample a hyperparameter value according to its configuration.
        
        Args:
            param_name (str): Name of the hyperparameter
            param_config (dict): Configuration with 'type', 'min', 'max', 'scale'
        
        Returns:
            Sampled value
        """
        param_type = param_config['type']
        min_val = param_config['min']
        max_val = param_config['max']
        scale = param_config.get('scale', 'linear')
        
        if scale == 'log':
            # Log scale sampling
            log_min = np.log(min_val)
            log_max = np.log(max_val)
            sampled_log = np.random.uniform(log_min, log_max)
            sampled = np.exp(sampled_log)
        else:
            # Linear scale sampling
            sampled = np.random.uniform(min_val, max_val)
        
        # Convert to appropriate type
        if param_type == 'int':
            sampled = int(np.round(sampled))
        elif param_type == 'float':
            sampled = float(sampled)
        
        return sampled
    
    def sample_config(self):
        """Sample a random configuration from search space."""
        config = {}
        for param_name, param_config in self.search_space.items():
            config[param_name] = self.sample_parameter(param_name, param_config)
        return config
    
    def record_evaluation(self, config, score, metadata=None):
        """
        Record an evaluation result.
        
        Args:
            config (dict): Hyperparameter configuration
            score (float): Validation score
            metadata (dict): Additional metadata (optional)
        """
        record = {
            'config': config.copy(),
            'score': score,
            'metadata': metadata or {}
        }
        self.history.append(record)
        
        # Update best
        if score > self.best_score:
            self.best_score = score
            self.best_config = config.copy()
    
    @abstractmethod
    def optimize(self, objective_function, n_iterations):
        """
        Run optimization.
        
        Args:
            objective_function: Function that takes config and returns score
            n_iterations (int): Number of iterations
        
        Returns:
            dict: Best configuration found
        """
        pass

