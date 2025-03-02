from typing import Dict, Any, Optional, Union, List, Tuple, Type, Callable
import numpy as np

from src.genetic_algorithm.base import (
    Chromosome, SelectionOperator, CrossoverOperator, MutationOperator, GeneticAlgorithm
)
from src.genetic_algorithm.selection import (
    RouletteWheelSelection, TournamentSelection, RankSelection, ElitismSelection
)
from src.genetic_algorithm.crossover import (
    SinglePointCrossover, TwoPointCrossover, UniformCrossover, BlendCrossover
)
from src.genetic_algorithm.mutation import (
    BitFlipMutation, SwapMutation, GaussianMutation, UniformMutation, InversionMutation, AdaptiveMutation
)
from src.genetic_algorithm.initialization import (
    PopulationInitializer, RandomInitializer, HeuristicInitializer, SeededInitializer, DiverseInitializer
)
from src.genetic_algorithm.config import GeneticAlgorithmConfig


class GeneticAlgorithmFactory:
    """
    Factory class for creating genetic algorithm components.
    
    This class creates genetic algorithm components based on configuration
    parameters, including selection operators, crossover operators, mutation
    operators, and initializers.
    """
    
    @staticmethod
    def create_selection_operator(
        config: GeneticAlgorithmConfig
    ) -> SelectionOperator:
        """
        Create a selection operator based on the configuration.
        
        Args:
            config: The genetic algorithm configuration.
            
        Returns:
            A selection operator.
            
        Raises:
            ValueError: If the selection method is not supported.
        """
        method = config.selection_method.lower()
        
        if method == 'roulette':
            return RouletteWheelSelection()
        elif method == 'tournament':
            return TournamentSelection(tournament_size=config.tournament_size)
        elif method == 'rank':
            return RankSelection(selection_pressure=config.selection_pressure)
        elif method == 'elitism':
            return ElitismSelection(elitism_rate=config.elitism_rate)
        else:
            raise ValueError(f"Unsupported selection method: {method}")
    
    @staticmethod
    def create_crossover_operator(
        config: GeneticAlgorithmConfig,
        chromosome_class: Type[Chromosome]
    ) -> CrossoverOperator:
        """
        Create a crossover operator based on the configuration.
        
        Args:
            config: The genetic algorithm configuration.
            chromosome_class: The class of the chromosomes to create.
            
        Returns:
            A crossover operator.
            
        Raises:
            ValueError: If the crossover method is not supported.
        """
        method = config.crossover_method.lower()
        
        if method == 'single_point':
            return SinglePointCrossover(chromosome_class=chromosome_class)
        elif method == 'two_point':
            return TwoPointCrossover(chromosome_class=chromosome_class)
        elif method == 'uniform':
            return UniformCrossover(
                chromosome_class=chromosome_class,
                swap_probability=config.uniform_crossover_probability
            )
        elif method == 'blend':
            return BlendCrossover(
                chromosome_class=chromosome_class,
                alpha=config.blend_alpha
            )
        else:
            raise ValueError(f"Unsupported crossover method: {method}")
    
    @staticmethod
    def create_mutation_operator(
        config: GeneticAlgorithmConfig
    ) -> MutationOperator:
        """
        Create a mutation operator based on the configuration.
        
        Args:
            config: The genetic algorithm configuration.
            
        Returns:
            A mutation operator.
            
        Raises:
            ValueError: If the mutation method is not supported.
        """
        method = config.mutation_method.lower()
        
        # Create base mutation operator
        if method == 'bit_flip':
            base_operator = BitFlipMutation(mutation_rate=config.mutation_probability)
        elif method == 'gaussian':
            base_operator = GaussianMutation(
                mutation_rate=config.mutation_probability,
                scale=config.mutation_scale
            )
        elif method == 'uniform':
            base_operator = UniformMutation(
                mutation_rate=config.mutation_probability,
                min_val=config.gene_min_value,
                max_val=config.gene_max_value
            )
        elif method == 'swap':
            base_operator = SwapMutation(mutation_rate=config.mutation_probability)
        elif method == 'inversion':
            base_operator = InversionMutation(mutation_rate=config.mutation_probability)
        else:
            raise ValueError(f"Unsupported mutation method: {method}")
        
        # Wrap with adaptive mutation if enabled
        if config.adaptive_mutation:
            return AdaptiveMutation(
                base_mutation_rate=config.mutation_probability,
                min_mutation_rate=config.min_mutation_probability,
                max_mutation_rate=config.max_mutation_probability,
                base_operator=base_operator
            )
        
        return base_operator
    
    @staticmethod
    def create_initializer(
        config: GeneticAlgorithmConfig,
        chromosome_class: Type[Chromosome],
        gene_shape: Tuple[int, ...],
        heuristic_function: Optional[Callable[..., np.ndarray]] = None,
        seed_chromosomes: Optional[List[Chromosome]] = None
    ) -> PopulationInitializer:
        """
        Create a population initializer based on the configuration.
        
        Args:
            config: The genetic algorithm configuration.
            chromosome_class: The class of the chromosomes to create.
            gene_shape: The shape of the genes for each chromosome.
            heuristic_function: A function that generates genes using a heuristic.
            seed_chromosomes: A list of seed chromosomes to use.
            
        Returns:
            A population initializer.
            
        Raises:
            ValueError: If the initialization method is not supported.
        """
        method = config.initialization_method.lower()
        
        if method == 'random':
            return RandomInitializer(
                chromosome_class=chromosome_class,
                population_size=config.population_size,
                gene_shape=gene_shape,
                gene_type=config.gene_type,
                min_val=config.gene_min_value,
                max_val=config.gene_max_value
            )
        elif method == 'heuristic':
            if heuristic_function is None:
                raise ValueError("Heuristic function must be provided for heuristic initialization")
            
            return HeuristicInitializer(
                chromosome_class=chromosome_class,
                population_size=config.population_size,
                heuristic_function=heuristic_function,
                random_proportion=config.heuristic_random_proportion
            )
        elif method == 'seeded':
            if seed_chromosomes is None or len(seed_chromosomes) == 0:
                raise ValueError("Seed chromosomes must be provided for seeded initialization")
            
            return SeededInitializer(
                chromosome_class=chromosome_class,
                population_size=config.population_size,
                seed_chromosomes=seed_chromosomes,
                mutation_rate=config.mutation_probability,
                mutation_scale=config.mutation_scale
            )
        elif method == 'diverse':
            return DiverseInitializer(
                chromosome_class=chromosome_class,
                population_size=config.population_size,
                gene_shape=gene_shape,
                gene_type=config.gene_type,
                min_val=config.gene_min_value,
                max_val=config.gene_max_value,
                diversity_threshold=config.diversity_threshold
            )
        else:
            raise ValueError(f"Unsupported initialization method: {method}")
    
    @staticmethod
    def create_genetic_algorithm(
        config: GeneticAlgorithmConfig,
        chromosome_class: Type[Chromosome],
        gene_shape: Tuple[int, ...],
        heuristic_function: Optional[Callable[..., np.ndarray]] = None,
        seed_chromosomes: Optional[List[Chromosome]] = None
    ) -> GeneticAlgorithm:
        """
        Create a genetic algorithm based on the configuration.
        
        Args:
            config: The genetic algorithm configuration.
            chromosome_class: The class of the chromosomes to create.
            gene_shape: The shape of the genes for each chromosome.
            heuristic_function: A function that generates genes using a heuristic.
            seed_chromosomes: A list of seed chromosomes to use.
            
        Returns:
            A genetic algorithm.
        """
        # Set random seed if provided
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # Create operators
        selection_operator = GeneticAlgorithmFactory.create_selection_operator(config)
        crossover_operator = GeneticAlgorithmFactory.create_crossover_operator(config, chromosome_class)
        mutation_operator = GeneticAlgorithmFactory.create_mutation_operator(config)
        
        # Create genetic algorithm
        ga = GeneticAlgorithm(
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            population_size=config.population_size,
            elite_count=config.elite_count if config.elitism else 0,
            max_generations=config.max_generations,
            target_fitness=config.target_fitness,
            fitness_stagnation_limit=config.fitness_stagnation_limit,
            verbose=config.verbose
        )
        
        # Create initializer
        initializer = GeneticAlgorithmFactory.create_initializer(
            config=config,
            chromosome_class=chromosome_class,
            gene_shape=gene_shape,
            heuristic_function=heuristic_function,
            seed_chromosomes=seed_chromosomes
        )
        
        # Set initializer
        ga.initializer = initializer
        
        return ga 