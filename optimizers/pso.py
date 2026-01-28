"""
Particle Swarm Optimization (PSO) for hyperparameter tuning.
"""
import numpy as np
from .base_optimizer import BaseOptimizer


class ParticleSwarmOptimizer(BaseOptimizer):
    """
    Particle Swarm Optimization for hyperparameter tuning.
    """
    
    def __init__(self, search_space, population_size=10, w=0.7, c1=1.5, c2=1.5, seed=42):
        """
        Initialize PSO optimizer.
        
        Args:
            search_space (dict): Search space definition
            population_size (int): Number of particles
            w (float): Inertia weight
            c1 (float): Cognitive coefficient
            c2 (float): Social coefficient
            seed (int): Random seed
        """
        super().__init__(search_space, seed)
        self.population_size = population_size
        self.w = w  # Inertia
        self.c1 = c1  # Cognitive
        self.c2 = c2  # Social
        self.name = "PSO"
        
        # Initialize particles
        self.particles = []
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_scores = []
        self.global_best_position = None
        self.global_best_score = -np.inf
        
        self._initialize_particles()
    
    def _initialize_particles(self):
        """Initialize particle positions and velocities."""
        param_names = list(self.search_space.keys())
        n_params = len(param_names)
        
        # Initialize particles
        for _ in range(self.population_size):
            config = self.sample_config()
            position = self._config_to_vector(config, param_names)
            self.particles.append(position)
            self.personal_best_positions.append(position.copy())
            self.personal_best_scores.append(-np.inf)
            
            # Initialize velocity (small random values)
            velocity = np.random.uniform(-0.1, 0.1, size=n_params)
            self.velocities.append(velocity)
    
    def _config_to_vector(self, config, param_names):
        """Convert config dict to parameter vector."""
        vector = []
        for param_name in param_names:
            param_config = self.search_space[param_name]
            value = config[param_name]
            
            # Normalize to [0, 1] range
            min_val = param_config['min']
            max_val = param_config['max']
            scale = param_config.get('scale', 'linear')
            
            if scale == 'log':
                # Log scale normalization
                log_min = np.log(min_val)
                log_max = np.log(max_val)
                log_val = np.log(value)
                normalized = (log_val - log_min) / (log_max - log_min) if log_max != log_min else 0.5
            else:
                # Linear normalization
                normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            
            vector.append(normalized)
        
        return np.array(vector)
    
    def _vector_to_config(self, vector, param_names):
        """Convert parameter vector to config dict."""
        config = {}
        for i, param_name in enumerate(param_names):
            param_config = self.search_space[param_name]
            normalized = np.clip(vector[i], 0.0, 1.0)
            
            min_val = param_config['min']
            max_val = param_config['max']
            scale = param_config.get('scale', 'linear')
            
            if scale == 'log':
                # Log scale denormalization
                log_min = np.log(min_val)
                log_max = np.log(max_val)
                log_val = log_min + normalized * (log_max - log_min)
                value = np.exp(log_val)
            else:
                # Linear denormalization
                value = min_val + normalized * (max_val - min_val)
            
            # Convert to appropriate type
            param_type = param_config['type']
            if param_type == 'int':
                value = int(np.round(value))
            elif param_type == 'float':
                value = float(value)
            
            config[param_name] = value
        
        return config
    
    def optimize(self, objective_function, n_iterations=10):
        """
        Run PSO optimization.
        
        Args:
            objective_function: Function(config, trial_number) -> score
            n_iterations (int): Number of PSO iterations
        
        Returns:
            dict: Best configuration found
        """
        print(f"\n{'='*60}")
        print(f"Starting PSO with {n_iterations} iterations, {self.population_size} particles")
        print(f"{'='*60}\n")
        
        param_names = list(self.search_space.keys())
        evaluation_counter = 0
        
        for iteration in range(n_iterations):
            print(f"\nPSO Iteration {iteration+1}/{n_iterations}")
            print("-" * 60)
            
            # Evaluate all particles
            for particle_idx in range(self.population_size):
                # Convert particle position to config
                config = self._vector_to_config(self.particles[particle_idx], param_names)
                
                print(f"  Particle {particle_idx+1}/{self.population_size}: {config}")
                
                # Evaluate
                score = objective_function(config, trial_number=evaluation_counter)
                evaluation_counter += 1
                
                print(f"  Validation Accuracy: {score:.4f}%\n")
                
                # Update personal best
                if score > self.personal_best_scores[particle_idx]:
                    self.personal_best_scores[particle_idx] = score
                    self.personal_best_positions[particle_idx] = self.particles[particle_idx].copy()
                
                # Update global best
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[particle_idx].copy()
                    self.best_config = config.copy()
                    self.best_score = score
                
                # Record evaluation
                self.record_evaluation(config, score, metadata={
                    'iteration': iteration+1,
                    'particle': particle_idx+1
                })
            
            # Update velocities and positions
            for particle_idx in range(self.population_size):
                r1 = np.random.random(len(param_names))
                r2 = np.random.random(len(param_names))
                
                # Velocity update
                cognitive = self.c1 * r1 * (self.personal_best_positions[particle_idx] - self.particles[particle_idx])
                social = self.c2 * r2 * (self.global_best_position - self.particles[particle_idx])
                self.velocities[particle_idx] = (
                    self.w * self.velocities[particle_idx] + cognitive + social
                )
                
                # Position update
                self.particles[particle_idx] += self.velocities[particle_idx]
                
                # Clip to [0, 1] bounds
                self.particles[particle_idx] = np.clip(self.particles[particle_idx], 0.0, 1.0)
            
            print(f"  Global Best Score: {self.global_best_score:.4f}%")
            print(f"  Global Best Config: {self.best_config}")
        
        print(f"\n{'='*60}")
        print(f"PSO Complete")
        print(f"Best Score: {self.best_score:.4f}%")
        print(f"Best Config: {self.best_config}")
        print(f"{'='*60}\n")
        
        return self.best_config

