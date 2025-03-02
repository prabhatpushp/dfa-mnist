import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Type
import random
from src.genetic_algorithm.base import Chromosome, CrossoverOperator


class SinglePointCrossover(CrossoverOperator):
    """
    Single-point crossover operator.
    
    This operator selects a random point in the chromosome and swaps the genes
    between the two parents at that point to create two offspring.
    """
    
    def __init__(self, chromosome_class: Type[Chromosome]):
        """
        Initialize the single-point crossover operator.
        
        Args:
            chromosome_class: The class of the chromosomes to create.
        """
        self.chromosome_class = chromosome_class
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform single-point crossover between two parent chromosomes.
        
        Args:
            parent1: The first parent chromosome.
            parent2: The second parent chromosome.
            
        Returns:
            A tuple containing two offspring chromosomes.
        """
        # Check if parents have the same gene length
        if parent1.genes.shape != parent2.genes.shape:
            raise ValueError("Parents must have the same gene length")
        
        # Get gene length
        gene_length = parent1.genes.size
        
        # If gene length is 1, just clone the parents
        if gene_length <= 1:
            return parent1.clone(), parent2.clone()
        
        # Select crossover point
        crossover_point = random.randint(1, gene_length - 1)
        
        # Create flattened copies of parent genes
        parent1_genes = parent1.genes.flatten()
        parent2_genes = parent2.genes.flatten()
        
        # Create offspring genes
        offspring1_genes = np.concatenate([
            parent1_genes[:crossover_point],
            parent2_genes[crossover_point:]
        ])
        
        offspring2_genes = np.concatenate([
            parent2_genes[:crossover_point],
            parent1_genes[crossover_point:]
        ])
        
        # Reshape offspring genes to match parent shape
        offspring1_genes = offspring1_genes.reshape(parent1.genes.shape)
        offspring2_genes = offspring2_genes.reshape(parent2.genes.shape)
        
        # Create offspring chromosomes
        offspring1 = self.chromosome_class(offspring1_genes)
        offspring2 = self.chromosome_class(offspring2_genes)
        
        return offspring1, offspring2


class TwoPointCrossover(CrossoverOperator):
    """
    Two-point crossover operator.
    
    This operator selects two random points in the chromosome and swaps the genes
    between the two parents between those points to create two offspring.
    """
    
    def __init__(self, chromosome_class: Type[Chromosome]):
        """
        Initialize the two-point crossover operator.
        
        Args:
            chromosome_class: The class of the chromosomes to create.
        """
        self.chromosome_class = chromosome_class
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform two-point crossover between two parent chromosomes.
        
        Args:
            parent1: The first parent chromosome.
            parent2: The second parent chromosome.
            
        Returns:
            A tuple containing two offspring chromosomes.
        """
        # Check if parents have the same gene length
        if parent1.genes.shape != parent2.genes.shape:
            raise ValueError("Parents must have the same gene length")
        
        # Get gene length
        gene_length = parent1.genes.size
        
        # If gene length is too small, use single-point crossover
        if gene_length <= 2:
            return SinglePointCrossover(self.chromosome_class).crossover(parent1, parent2)
        
        # Select crossover points
        point1, point2 = sorted(random.sample(range(1, gene_length), 2))
        
        # Create flattened copies of parent genes
        parent1_genes = parent1.genes.flatten()
        parent2_genes = parent2.genes.flatten()
        
        # Create offspring genes
        offspring1_genes = np.concatenate([
            parent1_genes[:point1],
            parent2_genes[point1:point2],
            parent1_genes[point2:]
        ])
        
        offspring2_genes = np.concatenate([
            parent2_genes[:point1],
            parent1_genes[point1:point2],
            parent2_genes[point2:]
        ])
        
        # Reshape offspring genes to match parent shape
        offspring1_genes = offspring1_genes.reshape(parent1.genes.shape)
        offspring2_genes = offspring2_genes.reshape(parent2.genes.shape)
        
        # Create offspring chromosomes
        offspring1 = self.chromosome_class(offspring1_genes)
        offspring2 = self.chromosome_class(offspring2_genes)
        
        return offspring1, offspring2


class UniformCrossover(CrossoverOperator):
    """
    Uniform crossover operator.
    
    This operator swaps each gene between the two parents with a certain
    probability to create two offspring.
    """
    
    def __init__(self, chromosome_class: Type[Chromosome], swap_probability: float = 0.5):
        """
        Initialize the uniform crossover operator.
        
        Args:
            chromosome_class: The class of the chromosomes to create.
            swap_probability: The probability of swapping each gene.
        """
        self.chromosome_class = chromosome_class
        self.swap_probability = swap_probability
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform uniform crossover between two parent chromosomes.
        
        Args:
            parent1: The first parent chromosome.
            parent2: The second parent chromosome.
            
        Returns:
            A tuple containing two offspring chromosomes.
        """
        # Check if parents have the same gene length
        if parent1.genes.shape != parent2.genes.shape:
            raise ValueError("Parents must have the same gene length")
        
        # Create copies of parent genes
        parent1_genes = parent1.genes.copy()
        parent2_genes = parent2.genes.copy()
        
        # Create mask for gene swapping
        mask = np.random.random(parent1_genes.shape) < self.swap_probability
        
        # Create offspring genes
        offspring1_genes = np.where(mask, parent2_genes, parent1_genes)
        offspring2_genes = np.where(mask, parent1_genes, parent2_genes)
        
        # Create offspring chromosomes
        offspring1 = self.chromosome_class(offspring1_genes)
        offspring2 = self.chromosome_class(offspring2_genes)
        
        return offspring1, offspring2


class BlendCrossover(CrossoverOperator):
    """
    Blend crossover operator (BLX-alpha).
    
    This operator is designed for real-valued genes. It creates offspring
    by blending the genes of the parents using a parameter alpha.
    """
    
    def __init__(self, chromosome_class: Type[Chromosome], alpha: float = 0.5):
        """
        Initialize the blend crossover operator.
        
        Args:
            chromosome_class: The class of the chromosomes to create.
            alpha: The blending parameter (typically between 0 and 1).
        """
        self.chromosome_class = chromosome_class
        self.alpha = alpha
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform blend crossover between two parent chromosomes.
        
        Args:
            parent1: The first parent chromosome.
            parent2: The second parent chromosome.
            
        Returns:
            A tuple containing two offspring chromosomes.
        """
        # Check if parents have the same gene length
        if parent1.genes.shape != parent2.genes.shape:
            raise ValueError("Parents must have the same gene length")
        
        # Create copies of parent genes
        parent1_genes = parent1.genes.copy()
        parent2_genes = parent2.genes.copy()
        
        # Calculate gene ranges
        gene_min = np.minimum(parent1_genes, parent2_genes)
        gene_max = np.maximum(parent1_genes, parent2_genes)
        gene_range = gene_max - gene_min
        
        # Calculate extended ranges
        extended_min = gene_min - self.alpha * gene_range
        extended_max = gene_max + self.alpha * gene_range
        
        # Create offspring genes
        offspring1_genes = np.random.uniform(extended_min, extended_max)
        offspring2_genes = np.random.uniform(extended_min, extended_max)
        
        # Create offspring chromosomes
        offspring1 = self.chromosome_class(offspring1_genes)
        offspring2 = self.chromosome_class(offspring2_genes)
        
        return offspring1, offspring2 