from typing import Dict, Any, Optional, Union, List, Tuple
import json
import os


class GeneticAlgorithmConfig:
    """
    Configuration class for genetic algorithms.
    
    This class stores and manages configuration parameters for genetic algorithms,
    including population size, selection method, crossover method, mutation method,
    and termination criteria.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the configuration with default values or provided values.
        
        Args:
            **kwargs: Configuration parameters to override defaults.
        """
        # Population parameters
        self.population_size = kwargs.get('population_size', 100)
        self.chromosome_length = kwargs.get('chromosome_length', 100)
        self.gene_type = kwargs.get('gene_type', 'float')  # 'float', 'int', or 'binary'
        self.gene_min_value = kwargs.get('gene_min_value', 0.0)
        self.gene_max_value = kwargs.get('gene_max_value', 1.0)
        
        # Selection parameters
        self.selection_method = kwargs.get('selection_method', 'tournament')  # 'roulette', 'tournament', 'rank', 'elitist'
        self.tournament_size = kwargs.get('tournament_size', 3)
        self.selection_pressure = kwargs.get('selection_pressure', 1.5)
        
        # Crossover parameters
        self.crossover_method = kwargs.get('crossover_method', 'single_point')  # 'single_point', 'two_point', 'uniform', 'blend'
        self.crossover_probability = kwargs.get('crossover_probability', 0.8)
        self.uniform_crossover_probability = kwargs.get('uniform_crossover_probability', 0.5)
        self.blend_alpha = kwargs.get('blend_alpha', 0.5)
        
        # Mutation parameters
        self.mutation_method = kwargs.get('mutation_method', 'gaussian')  # 'bit_flip', 'gaussian', 'uniform', 'swap', 'inversion'
        self.mutation_probability = kwargs.get('mutation_probability', 0.1)
        self.mutation_scale = kwargs.get('mutation_scale', 0.1)
        self.adaptive_mutation = kwargs.get('adaptive_mutation', False)
        self.min_mutation_probability = kwargs.get('min_mutation_probability', 0.01)
        self.max_mutation_probability = kwargs.get('max_mutation_probability', 0.5)
        
        # Elitism parameters
        self.elitism = kwargs.get('elitism', True)
        self.elite_count = kwargs.get('elite_count', 1)
        
        # Termination criteria
        self.max_generations = kwargs.get('max_generations', 100)
        self.target_fitness = kwargs.get('target_fitness', None)
        self.fitness_stagnation_limit = kwargs.get('fitness_stagnation_limit', 20)
        self.time_limit = kwargs.get('time_limit', None)  # in seconds
        
        # Initialization parameters
        self.initialization_method = kwargs.get('initialization_method', 'random')  # 'random', 'heuristic', 'seeded', 'diverse'
        self.random_seed = kwargs.get('random_seed', None)
        self.heuristic_random_proportion = kwargs.get('heuristic_random_proportion', 0.5)
        self.diversity_threshold = kwargs.get('diversity_threshold', 0.1)
        
        # Logging parameters
        self.verbose = kwargs.get('verbose', True)
        self.log_interval = kwargs.get('log_interval', 10)
        self.save_best_chromosome = kwargs.get('save_best_chromosome', True)
        self.save_metrics = kwargs.get('save_metrics', True)
        self.output_directory = kwargs.get('output_directory', './output')
        
        # Problem-specific parameters
        self.problem_specific = kwargs.get('problem_specific', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            A dictionary containing the configuration parameters.
        """
        return {
            # Population parameters
            'population_size': self.population_size,
            'chromosome_length': self.chromosome_length,
            'gene_type': self.gene_type,
            'gene_min_value': self.gene_min_value,
            'gene_max_value': self.gene_max_value,
            
            # Selection parameters
            'selection_method': self.selection_method,
            'tournament_size': self.tournament_size,
            'selection_pressure': self.selection_pressure,
            
            # Crossover parameters
            'crossover_method': self.crossover_method,
            'crossover_probability': self.crossover_probability,
            'uniform_crossover_probability': self.uniform_crossover_probability,
            'blend_alpha': self.blend_alpha,
            
            # Mutation parameters
            'mutation_method': self.mutation_method,
            'mutation_probability': self.mutation_probability,
            'mutation_scale': self.mutation_scale,
            'adaptive_mutation': self.adaptive_mutation,
            'min_mutation_probability': self.min_mutation_probability,
            'max_mutation_probability': self.max_mutation_probability,
            
            # Elitism parameters
            'elitism': self.elitism,
            'elite_count': self.elite_count,
            
            # Termination criteria
            'max_generations': self.max_generations,
            'target_fitness': self.target_fitness,
            'fitness_stagnation_limit': self.fitness_stagnation_limit,
            'time_limit': self.time_limit,
            
            # Initialization parameters
            'initialization_method': self.initialization_method,
            'random_seed': self.random_seed,
            'heuristic_random_proportion': self.heuristic_random_proportion,
            'diversity_threshold': self.diversity_threshold,
            
            # Logging parameters
            'verbose': self.verbose,
            'log_interval': self.log_interval,
            'save_best_chromosome': self.save_best_chromosome,
            'save_metrics': self.save_metrics,
            'output_directory': self.output_directory,
            
            # Problem-specific parameters
            'problem_specific': self.problem_specific
        }
    
    def save(self, filepath: str) -> None:
        """
        Save the configuration to a JSON file.
        
        Args:
            filepath: The path to the file to save the configuration to.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save configuration to file
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, filepath: str) -> 'GeneticAlgorithmConfig':
        """
        Load a configuration from a JSON file.
        
        Args:
            filepath: The path to the file to load the configuration from.
            
        Returns:
            A new configuration object with the loaded parameters.
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def update(self, **kwargs) -> None:
        """
        Update the configuration with new values.
        
        Args:
            **kwargs: Configuration parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in self.problem_specific:
                self.problem_specific[key] = value
            else:
                self.problem_specific[key] = value
    
    def __str__(self) -> str:
        """
        Get a string representation of the configuration.
        
        Returns:
            A string representation of the configuration.
        """
        return json.dumps(self.to_dict(), indent=4)
    
    def __repr__(self) -> str:
        """
        Get a string representation of the configuration.
        
        Returns:
            A string representation of the configuration.
        """
        return self.__str__()


class ConfigBuilder:
    """
    Builder class for creating genetic algorithm configurations.
    
    This class provides a fluent interface for creating genetic algorithm
    configurations with method chaining.
    """
    
    def __init__(self):
        """
        Initialize the configuration builder with default values.
        """
        self.config_params = {}
    
    def with_population(
        self,
        size: int = 100,
        chromosome_length: int = 100,
        gene_type: str = 'float',
        gene_min_value: float = 0.0,
        gene_max_value: float = 1.0
    ) -> 'ConfigBuilder':
        """
        Set population parameters.
        
        Args:
            size: The size of the population.
            chromosome_length: The length of each chromosome.
            gene_type: The type of the genes ('float', 'int', or 'binary').
            gene_min_value: The minimum value for the genes (for 'float' and 'int' types).
            gene_max_value: The maximum value for the genes (for 'float' and 'int' types).
            
        Returns:
            The builder instance for method chaining.
        """
        self.config_params.update({
            'population_size': size,
            'chromosome_length': chromosome_length,
            'gene_type': gene_type,
            'gene_min_value': gene_min_value,
            'gene_max_value': gene_max_value
        })
        return self
    
    def with_selection(
        self,
        method: str = 'tournament',
        tournament_size: int = 3,
        selection_pressure: float = 1.5
    ) -> 'ConfigBuilder':
        """
        Set selection parameters.
        
        Args:
            method: The selection method ('roulette', 'tournament', 'rank', 'elitist').
            tournament_size: The size of the tournament for tournament selection.
            selection_pressure: The selection pressure for rank selection.
            
        Returns:
            The builder instance for method chaining.
        """
        self.config_params.update({
            'selection_method': method,
            'tournament_size': tournament_size,
            'selection_pressure': selection_pressure
        })
        return self
    
    def with_crossover(
        self,
        method: str = 'single_point',
        probability: float = 0.8,
        uniform_probability: float = 0.5,
        blend_alpha: float = 0.5
    ) -> 'ConfigBuilder':
        """
        Set crossover parameters.
        
        Args:
            method: The crossover method ('single_point', 'two_point', 'uniform', 'blend').
            probability: The probability of performing crossover.
            uniform_probability: The probability of swapping each gene in uniform crossover.
            blend_alpha: The blending parameter for blend crossover.
            
        Returns:
            The builder instance for method chaining.
        """
        self.config_params.update({
            'crossover_method': method,
            'crossover_probability': probability,
            'uniform_crossover_probability': uniform_probability,
            'blend_alpha': blend_alpha
        })
        return self
    
    def with_mutation(
        self,
        method: str = 'gaussian',
        probability: float = 0.1,
        scale: float = 0.1,
        adaptive: bool = False,
        min_probability: float = 0.01,
        max_probability: float = 0.5
    ) -> 'ConfigBuilder':
        """
        Set mutation parameters.
        
        Args:
            method: The mutation method ('bit_flip', 'gaussian', 'uniform', 'swap', 'inversion').
            probability: The probability of mutating each gene.
            scale: The scale of the mutations.
            adaptive: Whether to use adaptive mutation.
            min_probability: The minimum mutation probability for adaptive mutation.
            max_probability: The maximum mutation probability for adaptive mutation.
            
        Returns:
            The builder instance for method chaining.
        """
        self.config_params.update({
            'mutation_method': method,
            'mutation_probability': probability,
            'mutation_scale': scale,
            'adaptive_mutation': adaptive,
            'min_mutation_probability': min_probability,
            'max_mutation_probability': max_probability
        })
        return self
    
    def with_elitism(self, enabled: bool = True, count: int = 1) -> 'ConfigBuilder':
        """
        Set elitism parameters.
        
        Args:
            enabled: Whether to use elitism.
            count: The number of elite chromosomes to preserve.
            
        Returns:
            The builder instance for method chaining.
        """
        self.config_params.update({
            'elitism': enabled,
            'elite_count': count
        })
        return self
    
    def with_termination(
        self,
        max_generations: int = 100,
        target_fitness: Optional[float] = None,
        stagnation_limit: int = 20,
        time_limit: Optional[float] = None
    ) -> 'ConfigBuilder':
        """
        Set termination criteria.
        
        Args:
            max_generations: The maximum number of generations.
            target_fitness: The target fitness to achieve.
            stagnation_limit: The number of generations without improvement before stopping.
            time_limit: The time limit in seconds.
            
        Returns:
            The builder instance for method chaining.
        """
        self.config_params.update({
            'max_generations': max_generations,
            'target_fitness': target_fitness,
            'fitness_stagnation_limit': stagnation_limit,
            'time_limit': time_limit
        })
        return self
    
    def with_initialization(
        self,
        method: str = 'random',
        random_seed: Optional[int] = None,
        heuristic_random_proportion: float = 0.5,
        diversity_threshold: float = 0.1
    ) -> 'ConfigBuilder':
        """
        Set initialization parameters.
        
        Args:
            method: The initialization method ('random', 'heuristic', 'seeded', 'diverse').
            random_seed: The random seed for reproducibility.
            heuristic_random_proportion: The proportion of the population to initialize randomly for heuristic initialization.
            diversity_threshold: The minimum distance between chromosomes for diverse initialization.
            
        Returns:
            The builder instance for method chaining.
        """
        self.config_params.update({
            'initialization_method': method,
            'random_seed': random_seed,
            'heuristic_random_proportion': heuristic_random_proportion,
            'diversity_threshold': diversity_threshold
        })
        return self
    
    def with_logging(
        self,
        verbose: bool = True,
        log_interval: int = 10,
        save_best_chromosome: bool = True,
        save_metrics: bool = True,
        output_directory: str = './output'
    ) -> 'ConfigBuilder':
        """
        Set logging parameters.
        
        Args:
            verbose: Whether to print log messages.
            log_interval: The interval (in generations) at which to log progress.
            save_best_chromosome: Whether to save the best chromosome.
            save_metrics: Whether to save metrics.
            output_directory: The directory to save output files to.
            
        Returns:
            The builder instance for method chaining.
        """
        self.config_params.update({
            'verbose': verbose,
            'log_interval': log_interval,
            'save_best_chromosome': save_best_chromosome,
            'save_metrics': save_metrics,
            'output_directory': output_directory
        })
        return self
    
    def with_problem_specific(self, **kwargs) -> 'ConfigBuilder':
        """
        Set problem-specific parameters.
        
        Args:
            **kwargs: Problem-specific parameters.
            
        Returns:
            The builder instance for method chaining.
        """
        self.config_params['problem_specific'] = kwargs
        return self
    
    def build(self) -> GeneticAlgorithmConfig:
        """
        Build the configuration.
        
        Returns:
            A new configuration object with the specified parameters.
        """
        return GeneticAlgorithmConfig(**self.config_params) 