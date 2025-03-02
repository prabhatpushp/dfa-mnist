import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional, Union, Callable


class Chromosome(ABC):
    """
    Abstract base class for chromosomes in the genetic algorithm.
    
    A chromosome represents a potential solution to the problem being solved.
    It contains genes that encode the solution and has a fitness value that
    indicates how good the solution is.
    """
    
    def __init__(self, genes: np.ndarray, fitness: float = 0.0):
        """
        Initialize a chromosome with genes and fitness.
        
        Args:
            genes: The genes of the chromosome.
            fitness: The fitness value of the chromosome.
        """
        self.genes = genes
        self.fitness = fitness
    
    @abstractmethod
    def calculate_fitness(self, *args, **kwargs) -> float:
        """
        Calculate the fitness of the chromosome.
        
        Returns:
            The fitness value of the chromosome.
        """
        pass
    
    @abstractmethod
    def clone(self) -> 'Chromosome':
        """
        Create a deep copy of the chromosome.
        
        Returns:
            A new chromosome with the same genes.
        """
        pass
    
    def __lt__(self, other: 'Chromosome') -> bool:
        """
        Compare chromosomes based on fitness for sorting.
        
        Args:
            other: Another chromosome to compare with.
            
        Returns:
            True if this chromosome has lower fitness than the other.
        """
        return self.fitness < other.fitness


class GeneticOperator(ABC):
    """
    Abstract base class for genetic operators.
    
    Genetic operators are used to modify chromosomes during the evolution process.
    """
    
    @abstractmethod
    def apply(self, *args, **kwargs) -> Any:
        """
        Apply the genetic operator.
        
        Returns:
            The result of applying the operator.
        """
        pass


class SelectionOperator(GeneticOperator):
    """
    Abstract base class for selection operators.
    
    Selection operators are used to select chromosomes from the population
    for reproduction based on their fitness.
    """
    
    @abstractmethod
    def select(self, population: List[Chromosome], count: int) -> List[Chromosome]:
        """
        Select chromosomes from the population.
        
        Args:
            population: The population to select from.
            count: The number of chromosomes to select.
            
        Returns:
            A list of selected chromosomes.
        """
        pass
    
    def apply(self, population: List[Chromosome], count: int) -> List[Chromosome]:
        """
        Apply the selection operator to the population.
        
        Args:
            population: The population to select from.
            count: The number of chromosomes to select.
            
        Returns:
            A list of selected chromosomes.
        """
        return self.select(population, count)


class CrossoverOperator(GeneticOperator):
    """
    Abstract base class for crossover operators.
    
    Crossover operators are used to combine the genes of two parent chromosomes
    to create offspring.
    """
    
    @abstractmethod
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Perform crossover between two parent chromosomes.
        
        Args:
            parent1: The first parent chromosome.
            parent2: The second parent chromosome.
            
        Returns:
            A tuple containing two offspring chromosomes.
        """
        pass
    
    def apply(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Apply the crossover operator to two parent chromosomes.
        
        Args:
            parent1: The first parent chromosome.
            parent2: The second parent chromosome.
            
        Returns:
            A tuple containing two offspring chromosomes.
        """
        return self.crossover(parent1, parent2)


class MutationOperator(GeneticOperator):
    """
    Abstract base class for mutation operators.
    
    Mutation operators are used to introduce random changes to chromosomes
    to maintain genetic diversity.
    """
    
    @abstractmethod
    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Perform mutation on a chromosome.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome.
        """
        pass
    
    def apply(self, chromosome: Chromosome) -> Chromosome:
        """
        Apply the mutation operator to a chromosome.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome.
        """
        return self.mutate(chromosome)


class Population:
    """
    Class representing a population of chromosomes.
    
    A population is a collection of chromosomes that evolve over time.
    """
    
    def __init__(self, chromosomes: List[Chromosome]):
        """
        Initialize a population with a list of chromosomes.
        
        Args:
            chromosomes: The chromosomes in the population.
        """
        self.chromosomes = chromosomes
    
    def get_best(self) -> Chromosome:
        """
        Get the chromosome with the highest fitness.
        
        Returns:
            The chromosome with the highest fitness.
        """
        return max(self.chromosomes, key=lambda x: x.fitness)
    
    def get_worst(self) -> Chromosome:
        """
        Get the chromosome with the lowest fitness.
        
        Returns:
            The chromosome with the lowest fitness.
        """
        return min(self.chromosomes, key=lambda x: x.fitness)
    
    def get_average_fitness(self) -> float:
        """
        Get the average fitness of the population.
        
        Returns:
            The average fitness of the population.
        """
        return np.mean([c.fitness for c in self.chromosomes])
    
    def get_fitness_std(self) -> float:
        """
        Get the standard deviation of fitness in the population.
        
        Returns:
            The standard deviation of fitness in the population.
        """
        return np.std([c.fitness for c in self.chromosomes])
    
    def __len__(self) -> int:
        """
        Get the size of the population.
        
        Returns:
            The number of chromosomes in the population.
        """
        return len(self.chromosomes)


class GeneticAlgorithm:
    """
    Class implementing the genetic algorithm.
    
    The genetic algorithm is an optimization algorithm inspired by the process
    of natural selection. It evolves a population of chromosomes over multiple
    generations to find an optimal solution.
    """
    
    def __init__(
        self,
        selection_operator: SelectionOperator,
        crossover_operator: CrossoverOperator,
        mutation_operator: MutationOperator,
        population_size: int,
        elite_count: int = 0,
        max_generations: int = 100,
        target_fitness: Optional[float] = None,
        fitness_stagnation_limit: int = 20,
        verbose: bool = False
    ):
        """
        Initialize the genetic algorithm.
        
        Args:
            selection_operator: The operator used for selection.
            crossover_operator: The operator used for crossover.
            mutation_operator: The operator used for mutation.
            population_size: The size of the population.
            elite_count: The number of elite chromosomes to preserve.
            max_generations: The maximum number of generations.
            target_fitness: The target fitness to achieve.
            fitness_stagnation_limit: The number of generations without improvement before stopping.
            verbose: Whether to print progress information.
        """
        self.selection_operator = selection_operator
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.population_size = population_size
        self.elite_count = elite_count
        self.max_generations = max_generations
        self.target_fitness = target_fitness
        self.fitness_stagnation_limit = fitness_stagnation_limit
        self.verbose = verbose
        
        self.current_generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_chromosome = None
        self.population = None
    
    @abstractmethod
    def initialize_population(self, *args, **kwargs) -> Population:
        """
        Initialize the population with random chromosomes.
        
        Returns:
            A new population of random chromosomes.
        """
        pass
    
    def evolve(self, *args, **kwargs) -> Chromosome:
        """
        Evolve the population to find an optimal solution.
        
        Returns:
            The best chromosome found.
        """
        # Initialize population if not already initialized
        if self.population is None:
            self.population = self.initialize_population(*args, **kwargs)
        
        # Evaluate initial population
        self._evaluate_population(*args, **kwargs)
        
        # Track best chromosome
        self.best_chromosome = self.population.get_best()
        self.best_fitness_history.append(self.best_chromosome.fitness)
        self.avg_fitness_history.append(self.population.get_average_fitness())
        
        if self.verbose:
            print(f"Generation 0: Best Fitness = {self.best_chromosome.fitness:.4f}, "
                  f"Avg Fitness = {self.avg_fitness_history[-1]:.4f}")
        
        # Evolution loop
        stagnation_counter = 0
        for generation in range(1, self.max_generations + 1):
            self.current_generation = generation
            
            # Create new population
            new_population = self._create_new_population()
            
            # Update population
            self.population = Population(new_population)
            
            # Evaluate new population
            self._evaluate_population(*args, **kwargs)
            
            # Update best chromosome
            current_best = self.population.get_best()
            if current_best.fitness > self.best_chromosome.fitness:
                self.best_chromosome = current_best
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Update history
            self.best_fitness_history.append(self.best_chromosome.fitness)
            self.avg_fitness_history.append(self.population.get_average_fitness())
            
            # Print progress for each generation if verbose
            if self.verbose:
                print(f"Generation {generation}: Best Fitness = {self.best_chromosome.fitness:.4f}, "
                      f"Avg Fitness = {self.avg_fitness_history[-1]:.4f}")
            
            # Check termination conditions
            if self.target_fitness is not None and self.best_chromosome.fitness >= self.target_fitness:
                if self.verbose:
                    print(f"Target fitness reached at generation {generation}")
                break
            
            if stagnation_counter >= self.fitness_stagnation_limit:
                if self.verbose:
                    print(f"Fitness stagnation detected at generation {generation}")
                break
        
        return self.best_chromosome
    
    def _evaluate_population(self, *args, **kwargs) -> None:
        """
        Evaluate the fitness of all chromosomes in the population.
        """
        for chromosome in self.population.chromosomes:
            chromosome.fitness = chromosome.calculate_fitness(*args, **kwargs)
    
    def _create_new_population(self) -> List[Chromosome]:
        """
        Create a new population using selection, crossover, and mutation.
        
        Returns:
            A list of chromosomes for the new population.
        """
        new_population = []
        
        # Elitism: preserve the best chromosomes
        if self.elite_count > 0:
            # Sort population by fitness (descending)
            sorted_population = sorted(self.population.chromosomes, key=lambda x: x.fitness, reverse=True)
            elites = sorted_population[:self.elite_count]
            # Add clones of elites to new population
            new_population.extend([elite.clone() for elite in elites])
        
        # Fill the rest of the population with offspring
        while len(new_population) < self.population_size:
            # Select parents
            parents = self.selection_operator.select(self.population.chromosomes, 2)
            
            # Perform crossover
            offspring1, offspring2 = self.crossover_operator.crossover(parents[0], parents[1])
            
            # Perform mutation
            offspring1 = self.mutation_operator.mutate(offspring1)
            offspring2 = self.mutation_operator.mutate(offspring2)
            
            # Add offspring to new population
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        return new_population 