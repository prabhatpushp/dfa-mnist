import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Type, Callable
import random
from src.genetic_algorithm.base import Chromosome, Population


class PopulationInitializer:
    """
    Base class for population initializers.
    
    Population initializers are responsible for creating the initial population
    of chromosomes for the genetic algorithm.
    """
    
    def __init__(self, chromosome_class: Type[Chromosome], population_size: int):
        """
        Initialize the population initializer.
        
        Args:
            chromosome_class: The class of the chromosomes to create.
            population_size: The size of the population to create.
        """
        self.chromosome_class = chromosome_class
        self.population_size = population_size
    
    def initialize(self, *args, **kwargs) -> Population:
        """
        Initialize a population of chromosomes.
        
        Returns:
            A new population of chromosomes.
        """
        chromosomes = self._create_chromosomes(*args, **kwargs)
        return Population(chromosomes)
    
    def _create_chromosomes(self, *args, **kwargs) -> List[Chromosome]:
        """
        Create a list of chromosomes.
        
        Returns:
            A list of chromosomes.
        """
        raise NotImplementedError("Subclasses must implement this method")


class RandomInitializer(PopulationInitializer):
    """
    Random population initializer.
    
    This initializer creates a population of chromosomes with random genes.
    """
    
    def __init__(
        self,
        chromosome_class: Type[Chromosome],
        population_size: int,
        gene_shape: Tuple[int, ...],
        gene_type: str = 'float',
        min_val: float = 0.0,
        max_val: float = 1.0
    ):
        """
        Initialize the random population initializer.
        
        Args:
            chromosome_class: The class of the chromosomes to create.
            population_size: The size of the population to create.
            gene_shape: The shape of the genes for each chromosome.
            gene_type: The type of the genes ('float', 'int', or 'binary').
            min_val: The minimum value for the genes (for 'float' and 'int' types).
            max_val: The maximum value for the genes (for 'float' and 'int' types).
        """
        super().__init__(chromosome_class, population_size)
        self.gene_shape = gene_shape
        self.gene_type = gene_type
        self.min_val = min_val
        self.max_val = max_val
    
    def _create_chromosomes(self, *args, **kwargs) -> List[Chromosome]:
        """
        Create a list of chromosomes with random genes.
        
        Returns:
            A list of chromosomes with random genes.
        """
        chromosomes = []
        
        for _ in range(self.population_size):
            # Create random genes based on type
            if self.gene_type == 'binary':
                genes = np.random.randint(0, 2, self.gene_shape)
            elif self.gene_type == 'int':
                genes = np.random.randint(
                    int(self.min_val),
                    int(self.max_val) + 1,
                    self.gene_shape
                )
            else:  # 'float'
                genes = np.random.uniform(
                    self.min_val,
                    self.max_val,
                    self.gene_shape
                )
            
            # Create chromosome
            chromosome = self.chromosome_class(genes)
            chromosomes.append(chromosome)
        
        return chromosomes


class HeuristicInitializer(PopulationInitializer):
    """
    Heuristic population initializer.
    
    This initializer creates a population of chromosomes using a heuristic
    function to generate genes that are likely to have good fitness.
    """
    
    def __init__(
        self,
        chromosome_class: Type[Chromosome],
        population_size: int,
        heuristic_function: Callable[..., np.ndarray],
        random_proportion: float = 0.5
    ):
        """
        Initialize the heuristic population initializer.
        
        Args:
            chromosome_class: The class of the chromosomes to create.
            population_size: The size of the population to create.
            heuristic_function: A function that generates genes using a heuristic.
            random_proportion: The proportion of the population to initialize randomly.
        """
        super().__init__(chromosome_class, population_size)
        self.heuristic_function = heuristic_function
        self.random_proportion = max(0.0, min(1.0, random_proportion))
    
    def _create_chromosomes(self, *args, **kwargs) -> List[Chromosome]:
        """
        Create a list of chromosomes using a heuristic function.
        
        Returns:
            A list of chromosomes with heuristically generated genes.
        """
        chromosomes = []
        
        # Calculate number of random and heuristic chromosomes
        num_random = int(self.population_size * self.random_proportion)
        num_heuristic = self.population_size - num_random
        
        # Create heuristic chromosomes
        for _ in range(num_heuristic):
            genes = self.heuristic_function(*args, **kwargs)
            chromosome = self.chromosome_class(genes)
            chromosomes.append(chromosome)
        
        # Create random chromosomes (if any)
        if num_random > 0:
            # Get gene shape from a heuristic chromosome
            gene_shape = chromosomes[0].genes.shape if chromosomes else None
            
            # If no heuristic chromosomes were created, get gene shape from heuristic function
            if gene_shape is None:
                sample_genes = self.heuristic_function(*args, **kwargs)
                gene_shape = sample_genes.shape
            
            # Create random initializer
            random_initializer = RandomInitializer(
                self.chromosome_class,
                num_random,
                gene_shape
            )
            
            # Add random chromosomes
            chromosomes.extend(random_initializer._create_chromosomes())
        
        return chromosomes


class SeededInitializer(PopulationInitializer):
    """
    Seeded population initializer.
    
    This initializer creates a population of chromosomes by seeding with known
    good solutions and applying mutations to create variations.
    """
    
    def __init__(
        self,
        chromosome_class: Type[Chromosome],
        population_size: int,
        seed_chromosomes: List[Chromosome],
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.1
    ):
        """
        Initialize the seeded population initializer.
        
        Args:
            chromosome_class: The class of the chromosomes to create.
            population_size: The size of the population to create.
            seed_chromosomes: A list of seed chromosomes to use.
            mutation_rate: The probability of mutating each gene.
            mutation_scale: The scale of the mutations.
        """
        super().__init__(chromosome_class, population_size)
        self.seed_chromosomes = seed_chromosomes
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
    
    def _create_chromosomes(self, *args, **kwargs) -> List[Chromosome]:
        """
        Create a list of chromosomes by seeding with known good solutions.
        
        Returns:
            A list of chromosomes seeded from known good solutions.
        """
        if not self.seed_chromosomes:
            raise ValueError("No seed chromosomes provided")
        
        chromosomes = []
        
        # Add seed chromosomes
        for seed in self.seed_chromosomes:
            chromosomes.append(seed.clone())
        
        # Fill the rest of the population with mutations of the seeds
        while len(chromosomes) < self.population_size:
            # Select a random seed
            seed = random.choice(self.seed_chromosomes)
            
            # Create a mutated copy
            mutated = seed.clone()
            
            # Apply mutation
            mutation_mask = np.random.random(mutated.genes.shape) < self.mutation_rate
            noise = np.random.normal(0, self.mutation_scale, mutated.genes.shape)
            mutated.genes[mutation_mask] += noise[mutation_mask]
            
            # Add to population
            chromosomes.append(mutated)
        
        return chromosomes


class DiverseInitializer(PopulationInitializer):
    """
    Diverse population initializer.
    
    This initializer creates a population of chromosomes with diverse genes
    to ensure good coverage of the search space.
    """
    
    def __init__(
        self,
        chromosome_class: Type[Chromosome],
        population_size: int,
        gene_shape: Tuple[int, ...],
        gene_type: str = 'float',
        min_val: float = 0.0,
        max_val: float = 1.0,
        diversity_threshold: float = 0.1
    ):
        """
        Initialize the diverse population initializer.
        
        Args:
            chromosome_class: The class of the chromosomes to create.
            population_size: The size of the population to create.
            gene_shape: The shape of the genes for each chromosome.
            gene_type: The type of the genes ('float', 'int', or 'binary').
            min_val: The minimum value for the genes (for 'float' and 'int' types).
            max_val: The maximum value for the genes (for 'float' and 'int' types).
            diversity_threshold: The minimum distance between chromosomes.
        """
        super().__init__(chromosome_class, population_size)
        self.gene_shape = gene_shape
        self.gene_type = gene_type
        self.min_val = min_val
        self.max_val = max_val
        self.diversity_threshold = diversity_threshold
    
    def _create_chromosomes(self, *args, **kwargs) -> List[Chromosome]:
        """
        Create a list of chromosomes with diverse genes.
        
        Returns:
            A list of chromosomes with diverse genes.
        """
        chromosomes = []
        max_attempts = 100  # Maximum number of attempts to find a diverse chromosome
        
        # Create random initializer
        random_initializer = RandomInitializer(
            self.chromosome_class,
            1,
            self.gene_shape,
            self.gene_type,
            self.min_val,
            self.max_val
        )
        
        # Add first chromosome
        first_chromosome = random_initializer._create_chromosomes()[0]
        chromosomes.append(first_chromosome)
        
        # Add remaining chromosomes
        while len(chromosomes) < self.population_size:
            # Try to find a diverse chromosome
            for _ in range(max_attempts):
                # Create a random chromosome
                candidate = random_initializer._create_chromosomes()[0]
                
                # Check if it's diverse enough
                if self._is_diverse_enough(candidate, chromosomes):
                    chromosomes.append(candidate)
                    break
            else:
                # If no diverse chromosome was found after max_attempts, just add a random one
                candidate = random_initializer._create_chromosomes()[0]
                chromosomes.append(candidate)
        
        return chromosomes
    
    def _is_diverse_enough(self, candidate: Chromosome, population: List[Chromosome]) -> bool:
        """
        Check if a candidate chromosome is diverse enough compared to the population.
        
        Args:
            candidate: The candidate chromosome.
            population: The current population.
            
        Returns:
            True if the candidate is diverse enough, False otherwise.
        """
        for chromosome in population:
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((candidate.genes.flatten() - chromosome.genes.flatten()) ** 2))
            
            # Check if distance is below threshold
            if distance < self.diversity_threshold:
                return False
        
        return True 