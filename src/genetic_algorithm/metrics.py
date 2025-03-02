import numpy as np
from typing import List, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
from src.genetic_algorithm.base import Chromosome, Population


class GeneticAlgorithmMetrics:
    """
    Class for tracking and analyzing metrics from a genetic algorithm run.
    
    This class collects and analyzes metrics such as fitness statistics,
    diversity, and convergence rate.
    """
    
    def __init__(self):
        """
        Initialize the metrics tracker.
        """
        # Fitness metrics
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.worst_fitness_history = []
        self.fitness_std_history = []
        
        # Diversity metrics
        self.gene_diversity_history = []
        self.phenotype_diversity_history = []
        
        # Convergence metrics
        self.convergence_rate_history = []
        self.stagnation_count_history = []
        
        # Generation counter
        self.generation_count = 0
    
    def update(self, population: Population, best_chromosome: Optional[Chromosome] = None) -> None:
        """
        Update metrics with the current population.
        
        Args:
            population: The current population.
            best_chromosome: The best chromosome found so far (may be from a previous generation).
        """
        # Increment generation counter
        self.generation_count += 1
        
        # Get current best chromosome
        current_best = population.get_best()
        
        # Update fitness metrics
        self.best_fitness_history.append(current_best.fitness)
        self.avg_fitness_history.append(population.get_average_fitness())
        self.worst_fitness_history.append(population.get_worst().fitness)
        self.fitness_std_history.append(population.get_fitness_std())
        
        # Update diversity metrics
        self.gene_diversity_history.append(self._calculate_gene_diversity(population))
        
        # Update convergence metrics
        if len(self.best_fitness_history) > 1:
            # Calculate improvement rate
            improvement = self.best_fitness_history[-1] - self.best_fitness_history[-2]
            self.convergence_rate_history.append(improvement)
            
            # Calculate stagnation count
            if improvement <= 0:
                stagnation_count = self.stagnation_count_history[-1] + 1 if self.stagnation_count_history else 1
            else:
                stagnation_count = 0
            self.stagnation_count_history.append(stagnation_count)
        else:
            self.convergence_rate_history.append(0)
            self.stagnation_count_history.append(0)
    
    def _calculate_gene_diversity(self, population: Population) -> float:
        """
        Calculate the genetic diversity of the population.
        
        Args:
            population: The population to analyze.
            
        Returns:
            A measure of genetic diversity (higher is more diverse).
        """
        if len(population.chromosomes) <= 1:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        chromosomes = population.chromosomes
        
        for i in range(len(chromosomes)):
            for j in range(i + 1, len(chromosomes)):
                # Calculate Euclidean distance between gene vectors
                distance = np.sqrt(np.sum((chromosomes[i].genes.flatten() - chromosomes[j].genes.flatten()) ** 2))
                distances.append(distance)
        
        # Return average distance
        return np.mean(distances) if distances else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the metrics.
        
        Returns:
            A dictionary containing summary statistics.
        """
        if not self.best_fitness_history:
            return {
                "generations": 0,
                "best_fitness": None,
                "avg_fitness": None,
                "improvement_rate": None,
                "diversity": None,
                "stagnation": None
            }
        
        return {
            "generations": self.generation_count,
            "best_fitness": self.best_fitness_history[-1],
            "avg_fitness": self.avg_fitness_history[-1],
            "improvement_rate": np.mean(self.convergence_rate_history) if self.convergence_rate_history else 0,
            "diversity": self.gene_diversity_history[-1],
            "stagnation": self.stagnation_count_history[-1]
        }
    
    def plot_fitness_history(self, title: str = "Fitness History", figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot the fitness history.
        
        Args:
            title: The title of the plot.
            figsize: The size of the figure.
            
        Returns:
            The matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        generations = range(1, self.generation_count + 1)
        
        ax.plot(generations, self.best_fitness_history, label="Best Fitness", color="green", linewidth=2)
        ax.plot(generations, self.avg_fitness_history, label="Average Fitness", color="blue", linewidth=1.5)
        ax.plot(generations, self.worst_fitness_history, label="Worst Fitness", color="red", linewidth=1)
        
        # Add standard deviation band around average
        if self.fitness_std_history:
            ax.fill_between(
                generations,
                [avg - std for avg, std in zip(self.avg_fitness_history, self.fitness_std_history)],
                [avg + std for avg, std in zip(self.avg_fitness_history, self.fitness_std_history)],
                alpha=0.2,
                color="blue",
                label="Fitness Std Dev"
            )
        
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        
        return fig
    
    def plot_diversity_history(self, title: str = "Diversity History", figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot the diversity history.
        
        Args:
            title: The title of the plot.
            figsize: The size of the figure.
            
        Returns:
            The matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        generations = range(1, self.generation_count + 1)
        
        ax.plot(generations, self.gene_diversity_history, label="Genetic Diversity", color="purple", linewidth=2)
        
        ax.set_xlabel("Generation")
        ax.set_ylabel("Diversity")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        
        return fig
    
    def plot_convergence_rate(self, title: str = "Convergence Rate", figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot the convergence rate.
        
        Args:
            title: The title of the plot.
            figsize: The size of the figure.
            
        Returns:
            The matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        generations = range(2, self.generation_count + 1)  # Start from 2 since first generation has no improvement
        
        ax.plot(generations, self.convergence_rate_history[1:], label="Improvement Rate", color="orange", linewidth=2)
        
        # Add moving average
        window_size = min(10, len(self.convergence_rate_history) - 1)
        if window_size > 0:
            moving_avg = np.convolve(
                self.convergence_rate_history[1:],
                np.ones(window_size) / window_size,
                mode="valid"
            )
            ax.plot(
                range(window_size + 1, self.generation_count + 1),
                moving_avg,
                label=f"{window_size}-Gen Moving Avg",
                color="brown",
                linewidth=1.5,
                linestyle="--"
            )
        
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness Improvement")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        
        return fig
    
    def plot_stagnation(self, title: str = "Stagnation Count", figsize: tuple = (10, 6)) -> plt.Figure:
        """
        Plot the stagnation count.
        
        Args:
            title: The title of the plot.
            figsize: The size of the figure.
            
        Returns:
            The matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        generations = range(1, self.generation_count + 1)
        
        ax.plot(generations, self.stagnation_count_history, label="Stagnation Count", color="red", linewidth=2)
        
        ax.set_xlabel("Generation")
        ax.set_ylabel("Generations without Improvement")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        
        return fig
    
    def plot_all(self, figsize: tuple = (15, 10)) -> plt.Figure:
        """
        Plot all metrics in a single figure.
        
        Args:
            figsize: The size of the figure.
            
        Returns:
            The matplotlib figure.
        """
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        generations = range(1, self.generation_count + 1)
        
        # Fitness history
        axs[0, 0].plot(generations, self.best_fitness_history, label="Best", color="green", linewidth=2)
        axs[0, 0].plot(generations, self.avg_fitness_history, label="Average", color="blue", linewidth=1.5)
        axs[0, 0].plot(generations, self.worst_fitness_history, label="Worst", color="red", linewidth=1)
        axs[0, 0].set_xlabel("Generation")
        axs[0, 0].set_ylabel("Fitness")
        axs[0, 0].set_title("Fitness History")
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle="--", alpha=0.7)
        
        # Diversity history
        axs[0, 1].plot(generations, self.gene_diversity_history, label="Genetic Diversity", color="purple", linewidth=2)
        axs[0, 1].set_xlabel("Generation")
        axs[0, 1].set_ylabel("Diversity")
        axs[0, 1].set_title("Diversity History")
        axs[0, 1].legend()
        axs[0, 1].grid(True, linestyle="--", alpha=0.7)
        
        # Convergence rate
        if len(generations) > 1:
            axs[1, 0].plot(
                range(2, self.generation_count + 1),
                self.convergence_rate_history[1:],
                label="Improvement",
                color="orange",
                linewidth=2
            )
            axs[1, 0].set_xlabel("Generation")
            axs[1, 0].set_ylabel("Fitness Improvement")
            axs[1, 0].set_title("Convergence Rate")
            axs[1, 0].legend()
            axs[1, 0].grid(True, linestyle="--", alpha=0.7)
        
        # Stagnation count
        axs[1, 1].plot(generations, self.stagnation_count_history, label="Stagnation", color="red", linewidth=2)
        axs[1, 1].set_xlabel("Generation")
        axs[1, 1].set_ylabel("Generations without Improvement")
        axs[1, 1].set_title("Stagnation Count")
        axs[1, 1].legend()
        axs[1, 1].grid(True, linestyle="--", alpha=0.7)
        
        plt.tight_layout()
        
        return fig


class GeneticAlgorithmLogger:
    """
    Class for logging genetic algorithm progress.
    
    This class logs information about the genetic algorithm's progress,
    including fitness statistics, diversity, and convergence rate.
    """
    
    def __init__(self, log_interval: int = 1, verbose: bool = True, log_file: Optional[str] = None):
        """
        Initialize the logger.
        
        Args:
            log_interval: The interval (in generations) at which to log progress.
            verbose: Whether to print log messages.
            log_file: Path to a file where logs will be written. If None, logs are only printed to console.
        """
        self.log_interval = log_interval
        self.verbose = verbose
        self.log_file = log_file
        self.metrics = GeneticAlgorithmMetrics()
        
        # Create log file if specified
        if self.log_file is not None:
            with open(self.log_file, 'w') as f:
                f.write("Generation,Best Fitness,Average Fitness,Worst Fitness,Fitness Std,Gene Diversity,Phenotype Diversity,Convergence Rate,Stagnation Count\n")
    
    def log_generation(
        self,
        generation: int,
        population: Population,
        best_chromosome: Optional[Chromosome] = None,
        elapsed_time: Optional[float] = None
    ) -> None:
        """
        Log information about the current generation.
        
        Args:
            generation: The current generation number.
            population: The current population.
            best_chromosome: The best chromosome found so far.
            elapsed_time: The time elapsed for this generation.
        """
        # Skip logging if not at log interval
        if generation % self.log_interval != 0:
            return
        
        # Update metrics
        self.metrics.update(population, best_chromosome)
        
        # Get current metrics
        best_fitness = self.metrics.best_fitness_history[-1]
        avg_fitness = self.metrics.avg_fitness_history[-1]
        worst_fitness = self.metrics.worst_fitness_history[-1]
        fitness_std = self.metrics.fitness_std_history[-1]
        gene_diversity = self.metrics.gene_diversity_history[-1]
        phenotype_diversity = self.metrics.phenotype_diversity_history[-1]
        convergence_rate = self.metrics.convergence_rate_history[-1]
        stagnation_count = self.metrics.stagnation_count_history[-1]
        
        # Create log message
        log_message = f"Generation {generation}: "
        log_message += f"Best Fitness = {best_fitness:.4f}, "
        log_message += f"Avg Fitness = {avg_fitness:.4f}, "
        log_message += f"Worst Fitness = {worst_fitness:.4f}, "
        log_message += f"Fitness Std = {fitness_std:.4f}, "
        log_message += f"Gene Diversity = {gene_diversity:.4f}, "
        log_message += f"Phenotype Diversity = {phenotype_diversity:.4f}, "
        log_message += f"Convergence Rate = {convergence_rate:.4f}, "
        log_message += f"Stagnation Count = {stagnation_count}"
        
        if elapsed_time is not None:
            log_message += f", Time = {elapsed_time:.2f}s"
        
        # Print log message if verbose
        if self.verbose:
            print(log_message)
        
        # Write to log file if specified
        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                f.write(f"{generation},{best_fitness:.4f},{avg_fitness:.4f},{worst_fitness:.4f},{fitness_std:.4f},{gene_diversity:.4f},{phenotype_diversity:.4f},{convergence_rate:.4f},{stagnation_count}\n")
    
    def log_summary(self) -> None:
        """
        Log a summary of the genetic algorithm run.
        """
        if not self.verbose:
            return
        
        summary = self.metrics.get_summary()
        
        print("\nGenetic Algorithm Summary:")
        print(f"Total Generations: {summary['generations']}")
        print(f"Best Fitness: {summary['best_fitness']:.4f}")
        print(f"Average Fitness: {summary['avg_fitness']:.4f}")
        print(f"Average Improvement Rate: {summary['improvement_rate']:.6f}")
        print(f"Final Diversity: {summary['diversity']:.4f}")
        print(f"Final Stagnation Count: {summary['stagnation']}")
    
    def get_metrics(self) -> GeneticAlgorithmMetrics:
        """
        Get the metrics object.
        
        Returns:
            The metrics object.
        """
        return self.metrics 