import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import random
from src.genetic_algorithm.base import Chromosome, SelectionOperator


class RouletteWheelSelection(SelectionOperator):
    """
    Roulette wheel selection operator.
    
    This operator selects chromosomes with probability proportional to their fitness.
    Chromosomes with higher fitness have a higher chance of being selected.
    """
    
    def __init__(self):
        """
        Initialize the roulette wheel selection operator.
        """
        pass
    
    def select(self, population: List[Chromosome], count: int) -> List[Chromosome]:
        """
        Select chromosomes from the population using roulette wheel selection.
        
        Args:
            population: The population to select from.
            count: The number of chromosomes to select.
            
        Returns:
            A list of selected chromosomes.
        """
        # Handle edge cases
        if not population:
            return []
        
        if count <= 0:
            return []
        
        # Get fitness values
        fitness_values = np.array([chromosome.fitness for chromosome in population])
        
        # Handle negative fitness values by shifting
        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            fitness_values = fitness_values - min_fitness + 1e-10
        
        # Handle all zero fitness values
        if np.sum(fitness_values) == 0:
            # If all fitness values are zero, select randomly
            return random.choices(population, k=count)
        
        # Calculate selection probabilities
        probabilities = fitness_values / np.sum(fitness_values)
        
        # Select chromosomes
        selected_indices = np.random.choice(
            len(population),
            size=count,
            replace=True,
            p=probabilities
        )
        
        return [population[i] for i in selected_indices]


class TournamentSelection(SelectionOperator):
    """
    Tournament selection operator.
    
    This operator selects chromosomes by running tournaments among randomly
    selected subsets of the population. The chromosome with the highest fitness
    in each tournament is selected.
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        Initialize the tournament selection operator.
        
        Args:
            tournament_size: The number of chromosomes in each tournament.
        """
        self.tournament_size = tournament_size
    
    def select(self, population: List[Chromosome], count: int) -> List[Chromosome]:
        """
        Select chromosomes from the population using tournament selection.
        
        Args:
            population: The population to select from.
            count: The number of chromosomes to select.
            
        Returns:
            A list of selected chromosomes.
        """
        # Handle edge cases
        if not population:
            return []
        
        if count <= 0:
            return []
        
        # Adjust tournament size if necessary
        tournament_size = min(self.tournament_size, len(population))
        
        # Select chromosomes
        selected = []
        for _ in range(count):
            # Randomly select chromosomes for the tournament
            tournament = random.sample(population, tournament_size)
            
            # Select the best chromosome from the tournament
            winner = max(tournament, key=lambda x: x.fitness)
            
            selected.append(winner)
        
        return selected


class RankSelection(SelectionOperator):
    """
    Rank selection operator.
    
    This operator selects chromosomes based on their rank in the population
    rather than their absolute fitness. This helps maintain selection pressure
    when fitness values are close together.
    """
    
    def __init__(self, selection_pressure: float = 1.5):
        """
        Initialize the rank selection operator.
        
        Args:
            selection_pressure: A value between 1.0 and 2.0 that controls the
                                selection pressure. Higher values favor higher-ranked
                                chromosomes more strongly.
        """
        self.selection_pressure = max(1.0, min(2.0, selection_pressure))
    
    def select(self, population: List[Chromosome], count: int) -> List[Chromosome]:
        """
        Select chromosomes from the population using rank selection.
        
        Args:
            population: The population to select from.
            count: The number of chromosomes to select.
            
        Returns:
            A list of selected chromosomes.
        """
        # Handle edge cases
        if not population:
            return []
        
        if count <= 0:
            return []
        
        # Sort population by fitness (ascending)
        sorted_population = sorted(population, key=lambda x: x.fitness)
        
        # Calculate rank probabilities
        n = len(sorted_population)
        ranks = np.arange(1, n + 1)
        
        # Linear ranking formula
        probabilities = (2 - self.selection_pressure) / n + 2 * ranks * (self.selection_pressure - 1) / (n * (n + 1))
        
        # Select chromosomes
        selected_indices = np.random.choice(
            n,
            size=count,
            replace=True,
            p=probabilities
        )
        
        return [sorted_population[i] for i in selected_indices]


class ElitismSelection(SelectionOperator):
    """
    Elitist selection operator.
    
    This operator selects the best chromosomes from the population.
    It ensures that the best solutions are preserved.
    """
    
    def __init__(self, elitism_rate: float = 0.1):
        """
        Initialize the elitist selection operator.
        
        Args:
            elitism_rate: The proportion of the population to select as elites.
                          Default is 0.1 (10%).
        """
        self.elitism_rate = elitism_rate
    
    def select(self, population: List[Chromosome], count: int) -> List[Chromosome]:
        """
        Select the best chromosomes from the population.
        
        Args:
            population: The population to select from.
            count: The number of chromosomes to select.
            
        Returns:
            A list of selected chromosomes.
        """
        # Handle edge cases
        if not population:
            return []
        
        if count <= 0:
            return []
        
        # Sort population by fitness (descending)
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Calculate number of elites based on elitism_rate
        n_elites = max(1, min(int(len(population) * self.elitism_rate), count))
        
        # Select the best chromosomes
        return sorted_population[:min(n_elites, len(sorted_population))] 