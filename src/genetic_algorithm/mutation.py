import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import random
from src.genetic_algorithm.base import Chromosome, MutationOperator


class BitFlipMutation(MutationOperator):
    """
    Bit flip mutation operator.
    
    This operator flips bits in a binary chromosome with a certain probability.
    It is suitable for binary-encoded chromosomes.
    """
    
    def __init__(self, mutation_rate: float = 0.01):
        """
        Initialize the bit flip mutation operator.
        
        Args:
            mutation_rate: The probability of flipping each bit.
        """
        self.mutation_rate = mutation_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform bit flip mutation on a chromosome.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome.
        """
        # Create a copy of the chromosome
        mutated = chromosome.clone()
        
        # Create mutation mask
        mutation_mask = np.random.random(mutated.genes.shape) < self.mutation_rate
        
        # Apply mutation (flip bits)
        mutated.genes[mutation_mask] = 1 - mutated.genes[mutation_mask]
        
        return mutated


class SwapMutation(MutationOperator):
    """
    Swap mutation operator.
    
    This operator swaps two randomly selected genes in the chromosome.
    It is suitable for permutation-encoded chromosomes.
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize the swap mutation operator.
        
        Args:
            mutation_rate: The probability of performing a swap.
        """
        self.mutation_rate = mutation_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform swap mutation on a chromosome.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome.
        """
        # Create a copy of the chromosome
        mutated = chromosome.clone()
        
        # Get gene length
        gene_length = mutated.genes.size
        
        # If gene length is too small, return the original chromosome
        if gene_length <= 1:
            return mutated
        
        # Determine if mutation should be applied
        if random.random() < self.mutation_rate:
            # Flatten genes for easier manipulation
            flat_genes = mutated.genes.flatten()
            
            # Select two random positions
            pos1, pos2 = random.sample(range(gene_length), 2)
            
            # Swap genes
            flat_genes[pos1], flat_genes[pos2] = flat_genes[pos2], flat_genes[pos1]
            
            # Reshape genes back to original shape
            mutated.genes = flat_genes.reshape(mutated.genes.shape)
        
        return mutated


class GaussianMutation(MutationOperator):
    """
    Gaussian mutation operator.
    
    This operator adds Gaussian noise to genes with a certain probability.
    It is suitable for real-valued chromosomes.
    """
    
    def __init__(self, mutation_rate: float = 0.1, scale: float = 0.1):
        """
        Initialize the Gaussian mutation operator.
        
        Args:
            mutation_rate: The probability of mutating each gene.
            scale: The standard deviation of the Gaussian noise.
        """
        self.mutation_rate = mutation_rate
        self.scale = scale
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform Gaussian mutation on a chromosome.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome.
        """
        # Create a copy of the chromosome
        mutated = chromosome.clone()
        
        # Create mutation mask
        mutation_mask = np.random.random(mutated.genes.shape) < self.mutation_rate
        
        # Apply mutation (add Gaussian noise)
        noise = np.random.normal(0, self.scale, mutated.genes.shape)
        mutated.genes[mutation_mask] += noise[mutation_mask]
        
        return mutated


class UniformMutation(MutationOperator):
    """
    Uniform mutation operator.
    
    This operator replaces genes with random values from a uniform distribution
    with a certain probability. It is suitable for real-valued chromosomes.
    """
    
    def __init__(self, mutation_rate: float = 0.1, min_val: float = 0.0, max_val: float = 1.0):
        """
        Initialize the uniform mutation operator.
        
        Args:
            mutation_rate: The probability of mutating each gene.
            min_val: The minimum value for the uniform distribution.
            max_val: The maximum value for the uniform distribution.
        """
        self.mutation_rate = mutation_rate
        self.min_val = min_val
        self.max_val = max_val
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform uniform mutation on a chromosome.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome.
        """
        # Create a copy of the chromosome
        mutated = chromosome.clone()
        
        # Create mutation mask
        mutation_mask = np.random.random(mutated.genes.shape) < self.mutation_rate
        
        # Apply mutation (replace with random values)
        random_values = np.random.uniform(self.min_val, self.max_val, mutated.genes.shape)
        mutated.genes[mutation_mask] = random_values[mutation_mask]
        
        return mutated


class InversionMutation(MutationOperator):
    """
    Inversion mutation operator.
    
    This operator inverts a randomly selected segment of the chromosome.
    It is suitable for permutation-encoded chromosomes.
    """
    
    def __init__(self, mutation_rate: float = 0.1):
        """
        Initialize the inversion mutation operator.
        
        Args:
            mutation_rate: The probability of performing an inversion.
        """
        self.mutation_rate = mutation_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform inversion mutation on a chromosome.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome.
        """
        # Create a copy of the chromosome
        mutated = chromosome.clone()
        
        # Get gene length
        gene_length = mutated.genes.size
        
        # If gene length is too small, return the original chromosome
        if gene_length <= 2:
            return mutated
        
        # Determine if mutation should be applied
        if random.random() < self.mutation_rate:
            # Flatten genes for easier manipulation
            flat_genes = mutated.genes.flatten()
            
            # Select two random positions
            pos1, pos2 = sorted(random.sample(range(gene_length), 2))
            
            # Invert segment
            segment = flat_genes[pos1:pos2+1]
            flat_genes[pos1:pos2+1] = segment[::-1]
            
            # Reshape genes back to original shape
            mutated.genes = flat_genes.reshape(mutated.genes.shape)
        
        return mutated


class AdaptiveMutation(MutationOperator):
    """
    Adaptive mutation operator.
    
    This operator adjusts the mutation rate based on the fitness of the chromosome
    relative to the population. Chromosomes with lower fitness have a higher
    mutation rate to encourage exploration.
    """
    
    def __init__(
        self,
        base_mutation_rate: float = 0.1,
        min_mutation_rate: float = 0.01,
        max_mutation_rate: float = 0.5,
        base_operator: MutationOperator = None
    ):
        """
        Initialize the adaptive mutation operator.
        
        Args:
            base_mutation_rate: The base mutation rate.
            min_mutation_rate: The minimum mutation rate.
            max_mutation_rate: The maximum mutation rate.
            base_operator: The base mutation operator to use.
        """
        self.base_mutation_rate = base_mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.base_operator = base_operator or GaussianMutation(mutation_rate=base_mutation_rate)
        
        # Population statistics
        self.population_avg_fitness = 0.0
        self.population_max_fitness = 0.0
    
    def update_population_stats(self, population_avg_fitness: float, population_max_fitness: float):
        """
        Update the population statistics.
        
        Args:
            population_avg_fitness: The average fitness of the population.
            population_max_fitness: The maximum fitness of the population.
        """
        self.population_avg_fitness = population_avg_fitness
        self.population_max_fitness = population_max_fitness
    
    def calculate_mutation_rate(self, chromosome: Chromosome) -> float:
        """
        Calculate the mutation rate for a chromosome based on its fitness.
        
        Args:
            chromosome: The chromosome to calculate the mutation rate for.
            
        Returns:
            The mutation rate for the chromosome.
        """
        # If population statistics are not available, use base rate
        if self.population_max_fitness == 0 or self.population_avg_fitness == 0:
            return self.base_mutation_rate
        
        # Calculate relative fitness (0 to 1)
        if self.population_max_fitness == self.population_avg_fitness:
            relative_fitness = 1.0
        else:
            relative_fitness = (chromosome.fitness - self.population_avg_fitness) / (
                self.population_max_fitness - self.population_avg_fitness
            )
            relative_fitness = max(0.0, min(1.0, relative_fitness))
        
        # Calculate mutation rate (higher for lower fitness)
        mutation_rate = self.max_mutation_rate - relative_fitness * (
            self.max_mutation_rate - self.min_mutation_rate
        )
        
        return mutation_rate
    
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform adaptive mutation on a chromosome.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome.
        """
        # Calculate mutation rate
        mutation_rate = self.calculate_mutation_rate(chromosome)
        
        # Update base operator mutation rate
        if hasattr(self.base_operator, 'mutation_rate'):
            self.base_operator.mutation_rate = mutation_rate
        
        # Apply base operator
        return self.base_operator.mutate(chromosome) 