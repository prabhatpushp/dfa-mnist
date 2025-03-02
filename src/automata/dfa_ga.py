import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Set, Type
import random
from src.automata.dfa import DFAChromosome, DFA, STATE_INITIAL, STATE_FINAL, STATE_TRANSITION, STATE_DEAD
from src.genetic_algorithm.base import GeneticAlgorithm, Population, Chromosome
from src.genetic_algorithm.initialization import PopulationInitializer
from src.genetic_algorithm.crossover import CrossoverOperator
from src.genetic_algorithm.mutation import MutationOperator


class DFAGeneticAlgorithm(GeneticAlgorithm):
    """
    Genetic algorithm for evolving DFAs for MNIST digit recognition.
    
    This class extends the base GeneticAlgorithm class with DFA-specific
    initialization and evaluation methods.
    """
    
    def __init__(
        self,
        n_states: int = 20,
        n_symbols: int = 256,
        chunk_size: Optional[Tuple[int, int]] = None,
        chunk_stride: Optional[Tuple[int, int]] = None,
        learning_rate: float = 0.1,
        enable_self_learning: bool = True,
        pattern_memory_size: int = 5,
        min_learning_rate: float = 0.05,
        max_learning_rate: float = 0.3,
        learning_rate_adaptation: float = 0.1,
        feature_importance_threshold: float = 0.2,
        confidence_threshold: float = 0.7,
        forget_factor: float = 0.05,
        **kwargs
    ):
        """
        Initialize the DFA genetic algorithm.
        
        Args:
            n_states: The number of states in the DFA.
            n_symbols: The number of symbols in the alphabet.
            chunk_size: The size of chunks to process.
            chunk_stride: The stride for chunk processing.
            learning_rate: The rate at which the algorithm learns from successful patterns.
            enable_self_learning: Whether to enable self-learning.
            pattern_memory_size: The number of successful patterns to remember.
            min_learning_rate: Minimum learning rate.
            max_learning_rate: Maximum learning rate.
            learning_rate_adaptation: Rate at which learning rate adapts.
            feature_importance_threshold: Threshold for considering a feature important.
            confidence_threshold: Threshold for considering a prediction confident.
            forget_factor: Rate at which unsuccessful patterns are forgotten.
            **kwargs: Additional arguments for the base GeneticAlgorithm.
        """
        super().__init__(**kwargs)
        
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.learning_rate = learning_rate
        self.enable_self_learning = enable_self_learning
        self.pattern_memory_size = pattern_memory_size
        
        # Advanced learning parameters
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.learning_rate_adaptation = learning_rate_adaptation
        self.feature_importance_threshold = feature_importance_threshold
        self.confidence_threshold = confidence_threshold
        self.forget_factor = forget_factor
        
        # Initialize pattern memory
        self.pattern_memory = {digit: [] for digit in range(10)}
        self.successful_transitions = {}
        self.unsuccessful_transitions = {}
        self.digit_state_mapping = {}
        self.feature_importance = {}
        self.critical_transitions = set()
        self.last_best_fitness = 0.0
        self.stagnation_count = 0
        self.generation_improvements = []
        
        # Transition confidence tracking
        self.transition_confidence = {}  # (from_state, to_state, digit) -> confidence
        
        # Feature importance for each digit
        self.digit_feature_importance = {digit: np.zeros((28, 28)) for digit in range(10)}
        
        # Initialize data attributes for evaluation
        self.current_X = None
        self.current_y = None
    
    def initialize_population(self, *args, **kwargs) -> Population:
        """
        Initialize the population with random DFA chromosomes.
        
        Returns:
            A new population of random DFA chromosomes.
        """
        # If an initializer is provided, use it
        if hasattr(self, 'initializer') and self.initializer is not None:
            return self.initializer.initialize(*args, **kwargs)
        
        # Otherwise, create random chromosomes
        chromosomes = []
        
        # Calculate gene size
        transition_size = self.n_states * self.n_symbols
        state_types_size = self.n_states
        final_mapping_size = 10 * self.n_states
        gene_size = transition_size + state_types_size + final_mapping_size
        
        for _ in range(self.population_size):
            # Create random genes
            genes = np.random.random(gene_size)
            
            # Create chromosome
            chromosome = DFAChromosome(
                genes=genes,
                n_states=self.n_states,
                n_symbols=self.n_symbols,
                chunk_size=self.chunk_size,
                chunk_stride=self.chunk_stride
            )
            
            chromosomes.append(chromosome)
        
        return Population(chromosomes)
    
    def _evaluate_chromosome(self, chromosome):
        """
        Evaluate a single chromosome.
        
        Args:
            chromosome: The chromosome to evaluate.
            
        Returns:
            Tuple of (chromosome, fitness).
        """
        fitness = chromosome.calculate_fitness(self.current_X, self.current_y)
        return chromosome, fitness
    
    def _evaluate_population(self, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> None:
        """
        Evaluate the fitness of all chromosomes in the population using parallel processing.
        
        Args:
            X: The input data (MNIST images).
            y: The target labels (MNIST digits).
        """
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        import multiprocessing
        
        # Store current data for evaluation
        self.current_X = X
        self.current_y = y
        
        # Determine if we can use multiprocessing
        can_use_multiprocessing = True
        try:
            # Check if we're in a context where multiprocessing is allowed
            multiprocessing.get_context()
        except:
            can_use_multiprocessing = False
        
        # Use parallel processing if possible
        if can_use_multiprocessing and len(self.population.chromosomes) > 10:
            # Determine number of workers (use at most 75% of available cores)
            num_workers = max(1, int(multiprocessing.cpu_count() * 0.75))
            
            # Fall back to sequential processing since multiprocessing has pickling issues
            for chromosome in self.population.chromosomes:
                chromosome.fitness = chromosome.calculate_fitness(X, y)
                
                # If self-learning is enabled, analyze successful patterns
                if self.enable_self_learning and chromosome.fitness > 0.5:
                    self._analyze_successful_patterns(chromosome, X, y)
        else:
            # Sequential processing
            for chromosome in self.population.chromosomes:
                chromosome.fitness = chromosome.calculate_fitness(X, y)
                
                # If self-learning is enabled, analyze successful patterns
                if self.enable_self_learning and chromosome.fitness > 0.5:
                    self._analyze_successful_patterns(chromosome, X, y)
    
    def _analyze_successful_patterns(self, chromosome: DFAChromosome, X: np.ndarray, y: np.ndarray) -> None:
        """
        Analyze successful patterns in a chromosome with enhanced learning and optimized performance.
        
        Args:
            chromosome: The chromosome to analyze.
            X: The input data (MNIST images).
            y: The target labels (MNIST digits).
        """
        # Convert chromosome to DFA
        dfa = chromosome.to_dfa()
        
        # Sample a subset of the data for analysis (to avoid excessive computation)
        # Use stratified sampling to ensure all digits are represented
        sample_indices = []
        for digit in range(10):
            digit_indices = np.where(y == digit)[0]
            if len(digit_indices) > 0:
                # Take up to 5 samples per digit (reduced from 10 for speed)
                digit_samples = np.random.choice(digit_indices, min(5, len(digit_indices)), replace=False)
                sample_indices.extend(digit_samples)
        
        # Shuffle the indices
        np.random.shuffle(sample_indices)
        
        # Limit total samples
        sample_indices = sample_indices[:min(50, len(sample_indices))]  # Reduced from 100 for speed
        
        # Process samples in batches for better performance
        batch_size = 10
        for batch_start in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[batch_start:batch_start + batch_size]
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            
            # Process each sample in the batch
            for i, (image, digit) in enumerate(zip(batch_X, batch_y)):
                # Reset DFA
                dfa.reset()
                
                # Process the image
                original_state_history = []
                symbol_history = []
                position_history = []  # Track position in image for feature importance
                
                # Process image based on chunk configuration
                if dfa.use_chunks:
                    # Get image dimensions
                    height, width = image.shape
                    
                    # Get chunk parameters
                    chunk_height, chunk_width = dfa.chunk_size
                    stride_height, stride_width = dfa.chunk_stride
                    
                    # Calculate number of chunks
                    n_chunks_height = max(1, (height - chunk_height) // stride_height + 1)
                    n_chunks_width = max(1, (width - chunk_width) // stride_width + 1)
                    
                    # Pre-compute chunk indices for better performance
                    chunk_indices = []
                    for i in range(n_chunks_height):
                        for j in range(n_chunks_width):
                            start_h = i * stride_height
                            end_h = min(start_h + chunk_height, height)
                            start_w = j * stride_width
                            end_w = min(start_w + chunk_width, width)
                            chunk_indices.append((start_h, end_h, start_w, end_w))
                    
                    # Process all chunks
                    for start_h, end_h, start_w, end_w in chunk_indices:
                        # Extract chunk
                        chunk = image[start_h:end_h, start_w:end_w]
                        
                        # Calculate average pixel value in chunk
                        avg_value = np.mean(chunk)
                        
                        # Convert to symbol
                        symbol = min(int(avg_value * (dfa.n_symbols - 1)), dfa.n_symbols - 1)
                        
                        # Record current state and symbol
                        current_state = dfa.current_state
                        original_state_history.append(current_state)
                        symbol_history.append(symbol)
                        position_history.append((start_h, start_w, end_h, end_w))
                        
                        # Transition to the next state
                        dfa.current_state = dfa.transition_matrix[dfa.current_state, symbol]
                else:
                    # Flatten the image and convert to symbols in one step
                    flat_image = image.flatten()
                    symbols = np.minimum((flat_image * (dfa.n_symbols - 1)).astype(np.int32), dfa.n_symbols - 1)
                    
                    # Process in batches for better performance
                    pixel_batch_size = 64  # Process 64 pixels at a time
                    
                    for i in range(0, len(symbols), pixel_batch_size):
                        pixel_batch = symbols[i:i+pixel_batch_size]
                        
                        for pixel_idx, symbol in enumerate(pixel_batch, start=i):
                            # Calculate position in original image
                            row = pixel_idx // 28
                            col = pixel_idx % 28
                            
                            # Record current state, symbol, and position
                            current_state = dfa.current_state
                            original_state_history.append(current_state)
                            symbol_history.append(symbol)
                            position_history.append((row, col, row+1, col+1))  # Single pixel position
                            
                            # Transition to the next state
                            dfa.current_state = dfa.transition_matrix[dfa.current_state, symbol]
                
                # Check if the prediction is correct
                predicted_digit, confidence_scores = dfa.classify(image)
                
                # Calculate confidence for this prediction
                digit_confidence = confidence_scores[digit] if digit < len(confidence_scores) else 0
                
                if predicted_digit == digit and digit_confidence > self.confidence_threshold:
                    # This is a successful pattern - store it in memory
                    if len(self.pattern_memory[digit]) < self.pattern_memory_size:
                        self.pattern_memory[digit].append(image)
                    
                    # Record successful state transitions
                    final_state = dfa.current_state
                    
                    # Store the mapping between digit and final state
                    if digit not in self.digit_state_mapping:
                        self.digit_state_mapping[digit] = {}
                    
                    if final_state not in self.digit_state_mapping[digit]:
                        self.digit_state_mapping[digit][final_state] = 0
                    
                    self.digit_state_mapping[digit][final_state] += 1
                    
                    # Store successful transitions with confidence
                    # Process transitions in batches for better performance
                    transition_batch_size = 50
                    for i in range(0, len(original_state_history) - 1, transition_batch_size):
                        batch_end = min(i + transition_batch_size, len(original_state_history) - 1)
                        
                        for j in range(i, batch_end):
                            from_state = original_state_history[j]
                            to_state = original_state_history[j + 1]
                            symbol = symbol_history[j]
                            pos = position_history[j]
                            
                            transition_key = (from_state, to_state, digit)
                            
                            # Update transition count
                            if transition_key not in self.successful_transitions:
                                self.successful_transitions[transition_key] = 0
                            
                            self.successful_transitions[transition_key] += 1
                            
                            # Update transition confidence
                            if transition_key not in self.transition_confidence:
                                self.transition_confidence[transition_key] = 0.5  # Initial confidence
                            
                            # Increase confidence based on digit confidence
                            self.transition_confidence[transition_key] = min(
                                1.0, 
                                self.transition_confidence[transition_key] + 0.1 * digit_confidence
                            )
                            
                            # Update feature importance for this digit
                            # For each position in the image that led to a successful transition
                            start_h, start_w, end_h, end_w = pos
                            
                            # Increase importance of this region
                            importance_increment = 0.05 * digit_confidence
                            self.digit_feature_importance[digit][start_h:end_h, start_w:end_w] += importance_increment
                    
                    # Identify critical transitions (those that significantly impact classification)
                    # A transition is critical if it leads directly to the final state or
                    # if it's part of a sequence that consistently leads to correct classification
                    if len(original_state_history) > 1:
                        # Last transition is often critical
                        last_from = original_state_history[-2]
                        last_to = original_state_history[-1]
                        last_transition = (last_from, last_to, digit)
                        
                        self.critical_transitions.add(last_transition)
                        
                        # Also consider transitions with high confidence as critical
                        # Only check a subset of transitions for performance
                        check_indices = np.linspace(0, len(original_state_history) - 2, 10, dtype=int)
                        for idx in check_indices:
                            from_state = original_state_history[idx]
                            to_state = original_state_history[idx + 1]
                            trans_key = (from_state, to_state, digit)
                            
                            if trans_key in self.transition_confidence and self.transition_confidence[trans_key] > 0.8:
                                self.critical_transitions.add(trans_key)
                else:
                    # This is an unsuccessful pattern - record it to avoid in future
                    # Only process a subset of transitions for performance
                    check_indices = np.linspace(0, len(original_state_history) - 2, 10, dtype=int)
                    for idx in check_indices:
                        from_state = original_state_history[idx]
                        to_state = original_state_history[idx + 1]
                        
                        transition_key = (from_state, to_state, digit)
                        
                        # Update unsuccessful transition count
                        if transition_key not in self.unsuccessful_transitions:
                            self.unsuccessful_transitions[transition_key] = 0
                        
                        self.unsuccessful_transitions[transition_key] += 1
                        
                        # Decrease confidence in this transition
                        if transition_key in self.transition_confidence:
                            self.transition_confidence[transition_key] = max(
                                0.1,
                                self.transition_confidence[transition_key] - 0.05
                            )
        
        # Normalize feature importance maps
        for digit in range(10):
            max_importance = np.max(self.digit_feature_importance[digit])
            if max_importance > 0:
                self.digit_feature_importance[digit] = self.digit_feature_importance[digit] / max_importance
    
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
            
            # Apply self-learning if enabled
            if self.enable_self_learning:
                offspring1 = self._apply_self_learning(offspring1)
                offspring2 = self._apply_self_learning(offspring2)
            
            # Add offspring to new population
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        # Update mutation rates if using adaptive mutation
        if isinstance(self.mutation_operator, AdaptiveDFAMutation):
            self.mutation_operator.update_rates(new_population)
        
        return new_population
    
    def _apply_self_learning(self, chromosome: DFAChromosome) -> DFAChromosome:
        """
        Apply enhanced self-learning to a chromosome with optimized performance.
        
        Args:
            chromosome: The chromosome to apply self-learning to.
            
        Returns:
            The modified chromosome.
        """
        # Skip if no successful patterns have been found
        if not self.successful_transitions or not self.digit_state_mapping:
            return chromosome
        
        # Create a copy of the chromosome
        modified = chromosome.clone()
        
        # Convert to DFA to modify its structure
        dfa = modified.to_dfa()
        
        # Calculate gene sections
        transition_size = self.n_states * self.n_symbols
        state_types_size = self.n_states
        
        # Adjust learning rate based on fitness improvement
        if self.best_chromosome is not None:
            current_best_fitness = self.best_chromosome.fitness
            
            if current_best_fitness > self.last_best_fitness:
                # Fitness improved - increase learning rate
                self.learning_rate = min(
                    self.max_learning_rate,
                    self.learning_rate * (1 + self.learning_rate_adaptation)
                )
                self.stagnation_count = 0
                self.generation_improvements.append(self.current_generation)
            else:
                # Fitness stagnated - decrease learning rate
                self.stagnation_count += 1
                if self.stagnation_count > 3:
                    self.learning_rate = max(
                        self.min_learning_rate,
                        self.learning_rate * (1 - self.learning_rate_adaptation)
                    )
            
            self.last_best_fitness = current_best_fitness
        
        # Optimize: Pre-filter transitions for faster processing
        # Only consider transitions with sufficient confidence and count
        filtered_transitions = {
            key: count for key, count in self.successful_transitions.items()
            if (key in self.transition_confidence and 
                self.transition_confidence[key] > 0.6 and
                count > 2)
        }
        
        # Apply successful transitions with confidence-based probability
        # Process in batches for better performance
        transition_items = list(filtered_transitions.items())
        batch_size = 100
        
        for batch_start in range(0, len(transition_items), batch_size):
            batch_end = min(batch_start + batch_size, len(transition_items))
            batch = transition_items[batch_start:batch_end]
            
            for (from_state, to_state, digit), count in batch:
                # Get confidence for this transition
                confidence = self.transition_confidence.get((from_state, to_state, digit), 0.5)
                
                # Calculate learning probability based on count, confidence, and learning rate
                learning_prob = self.learning_rate * min(1.0, count / 10) * confidence
                
                # Apply with higher probability for critical transitions
                if (from_state, to_state, digit) in self.critical_transitions:
                    learning_prob *= 2.0
                
                # Check if this transition is also in unsuccessful transitions
                unsuccessful_count = self.unsuccessful_transitions.get((from_state, to_state, digit), 0)
                
                # If more unsuccessful than successful, reduce probability
                if unsuccessful_count > count:
                    learning_prob *= 0.5
                
                # Apply transition with calculated probability
                if random.random() < learning_prob:
                    # For critical transitions, try to find the actual symbol that triggered it
                    # by looking at feature importance
                    if (from_state, to_state, digit) in self.critical_transitions and np.max(self.digit_feature_importance[digit]) > 0:
                        # Find important symbols for this digit
                        important_features = self.digit_feature_importance[digit] > self.feature_importance_threshold
                        
                        if np.any(important_features):
                            # Choose a symbol that corresponds to an important feature
                            # This is a simplified approach - in practice, you'd need to map
                            # feature importance back to symbols more precisely
                            symbol = random.randint(max(1, self.n_symbols // 2), self.n_symbols - 1)
                        else:
                            # Fallback to random symbol
                            symbol = random.randint(0, self.n_symbols - 1)
                    else:
                        # Use random symbol for non-critical transitions
                        symbol = random.randint(0, self.n_symbols - 1)
                    
                    # Update transition matrix
                    dfa.transition_matrix[from_state, symbol] = to_state
        
        # Apply successful final state mappings with confidence weighting
        # Process in batches for better performance
        digit_items = list(self.digit_state_mapping.items())
        
        for digit, state_counts in digit_items:
            if not state_counts:
                continue
            
            # Find the most common final state for this digit
            most_common_state = max(state_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate confidence for this mapping
            mapping_confidence = state_counts[most_common_state] / sum(state_counts.values())
            
            # Increase the mapping value for this digit and state based on confidence
            mapping_increment = self.learning_rate * mapping_confidence
            dfa.final_state_mapping[digit, most_common_state] += mapping_increment
            
            # Mark the state as final with probability based on confidence
            if random.random() < self.learning_rate * mapping_confidence:
                dfa.state_types[most_common_state] = STATE_FINAL
        
        # Forget unsuccessful patterns - use vectorized operations for speed
        if self.unsuccessful_transitions:
            # Convert to numpy arrays for vectorized operations
            keys = list(self.unsuccessful_transitions.keys())
            counts = np.array([self.unsuccessful_transitions[k] for k in keys])
            
            # Apply forget factor
            new_counts = np.maximum(0, counts - self.forget_factor * counts)
            
            # Update dictionary
            for i, key in enumerate(keys):
                if new_counts[i] < 0.5:
                    del self.unsuccessful_transitions[key]
                else:
                    self.unsuccessful_transitions[key] = new_counts[i]
        
        # Update the chromosome's genes from the modified DFA
        # Transition matrix
        modified.genes[:transition_size] = dfa.transition_matrix.flatten()
        
        # State types
        state_types_normalized = dfa.state_types.astype(np.float32) / 3.0  # Normalize to [0, 1]
        modified.genes[transition_size:transition_size + state_types_size] = state_types_normalized
        
        # Final state mapping
        modified.genes[transition_size + state_types_size:] = dfa.final_state_mapping.flatten()
        
        # Clear the DFA cache to force recalculation
        modified._dfa = None
        
        return modified
    
    def evolve(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, X_test: np.ndarray = None, y_test: np.ndarray = None, eval_interval: int = 5, *args, **kwargs) -> Chromosome:
        """
        Evolve the population to find an optimal solution with optimized performance.
        
        Args:
            X: The input data (MNIST images).
            y: The target labels (MNIST digits).
            X_val: Validation data (optional).
            y_val: Validation labels (optional).
            X_test: Test data (optional).
            y_test: Test labels (optional).
            eval_interval: Interval (in generations) to evaluate on validation/test sets.
            
        Returns:
            The best chromosome found.
        """
        from tqdm import tqdm
        import time
        
        # Initialize population if not already initialized
        if self.population is None:
            self.population = self.initialize_population(X, y, *args, **kwargs)
        
        # Evaluate initial population
        try:
            self._evaluate_population(X, y, *args, **kwargs)
        except Exception as e:
            print(f"Warning: Error during population evaluation: {str(e)}")
            print("Falling back to sequential evaluation...")
            # Fallback to direct evaluation
            for chromosome in self.population.chromosomes:
                chromosome.fitness = chromosome.calculate_fitness(X, y)
                
                # If self-learning is enabled, analyze successful patterns
                if self.enable_self_learning and chromosome.fitness > 0.5:
                    try:
                        self._analyze_successful_patterns(chromosome, X, y)
                    except Exception as e2:
                        print(f"Warning: Error during pattern analysis: {str(e2)}")
        
        # Track best chromosome
        self.best_chromosome = self.population.get_best()
        self.best_fitness_history.append(self.best_chromosome.fitness)
        self.avg_fitness_history.append(self.population.get_average_fitness())
        
        # Initialize progress bar
        progress_bar = tqdm(total=self.max_generations, desc="Evolving DFA", unit="gen")
        
        # Print initial stats
        if self.verbose:
            print(f"\nGeneration 0: Best Fitness = {self.best_chromosome.fitness:.4f}, "
                  f"Avg Fitness = {self.avg_fitness_history[-1]:.4f}")
            
            # Evaluate on validation and test sets if provided
            if X_val is not None and y_val is not None:
                val_accuracy = self._evaluate_accuracy(self.best_chromosome, X_val, y_val)
                print(f"Validation Accuracy: {val_accuracy:.2%}")
            
            if X_test is not None and y_test is not None:
                test_accuracy = self._evaluate_accuracy(self.best_chromosome, X_test, y_test)
                print(f"Test Accuracy: {test_accuracy:.2%}")
        
        # Evolution loop
        stagnation_counter = 0
        start_time = time.time()
        
        # Prepare for early stopping
        best_val_accuracy = 0.0
        val_accuracy_history = []
        patience = 5  # Number of generations to wait for improvement
        
        # Track diversity for adaptive mutation
        gene_diversity_history = []
        
        for generation in range(1, self.max_generations + 1):
            self.current_generation = generation
            
            # Create new population
            new_population = self._create_new_population()
            
            # Update population
            self.population = Population(new_population)
            
            # Evaluate new population
            try:
                self._evaluate_population(X, y, *args, **kwargs)
            except Exception as e:
                print(f"Warning: Error during population evaluation in generation {generation}: {str(e)}")
                print("Falling back to sequential evaluation...")
                # Fallback to direct evaluation
                for chromosome in self.population.chromosomes:
                    chromosome.fitness = chromosome.calculate_fitness(X, y)
                    
                    # If self-learning is enabled, analyze successful patterns
                    if self.enable_self_learning and chromosome.fitness > 0.5:
                        try:
                            self._analyze_successful_patterns(chromosome, X, y)
                        except Exception as e2:
                            print(f"Warning: Error during pattern analysis: {str(e2)}")
            
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
            
            # Calculate gene diversity
            gene_diversity = np.mean(np.std([c.genes for c in self.population.chromosomes], axis=0))
            gene_diversity_history.append(gene_diversity)
            
            # Update progress bar
            elapsed_time = time.time() - start_time
            progress_bar.set_postfix({
                'best': f"{self.best_chromosome.fitness:.4f}",
                'avg': f"{self.avg_fitness_history[-1]:.4f}",
                'stag': stagnation_counter,
                'time': f"{elapsed_time:.1f}s"
            })
            progress_bar.update(1)
            
            # Print detailed stats at intervals or if verbose
            if self.verbose and (generation % eval_interval == 0 or generation == self.max_generations):
                print(f"\nGeneration {generation}: Best Fitness = {self.best_chromosome.fitness:.4f}, "
                      f"Avg Fitness = {self.avg_fitness_history[-1]:.4f}")
                
                # Calculate and print diversity metrics
                print(f"Gene Diversity: {gene_diversity:.4f}")
                
                # Evaluate on validation and test sets if provided
                val_accuracy = None
                if X_val is not None and y_val is not None:
                    val_accuracy = self._evaluate_accuracy(self.best_chromosome, X_val, y_val)
                    val_accuracy_history.append(val_accuracy)
                    print(f"Validation Accuracy: {val_accuracy:.2%}")
                    
                    # Update best validation accuracy
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        stagnation_counter = 0  # Reset stagnation counter if validation improves
                
                if X_test is not None and y_test is not None:
                    test_accuracy = self._evaluate_accuracy(self.best_chromosome, X_test, y_test)
                    print(f"Test Accuracy: {test_accuracy:.2%}")
                
                # Print mutation rates if using adaptive mutation
                if isinstance(self.mutation_operator, AdaptiveDFAMutation):
                    print(f"Mutation Rates - Overall: {self.mutation_operator.mutation_rate:.4f}, "
                          f"Transition: {self.mutation_operator.transition_mutation_rate:.4f}, "
                          f"State Type: {self.mutation_operator.state_type_mutation_rate:.4f}, "
                          f"Final Mapping: {self.mutation_operator.final_mapping_mutation_rate:.4f}")
                
                # Adjust learning parameters based on performance
                if val_accuracy is not None:
                    # If validation accuracy is improving, increase learning rate
                    if len(val_accuracy_history) > 1 and val_accuracy > val_accuracy_history[-2]:
                        self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.1)
                    # If validation accuracy is decreasing, decrease learning rate
                    elif len(val_accuracy_history) > 1 and val_accuracy < val_accuracy_history[-2]:
                        self.learning_rate = max(self.min_learning_rate, self.learning_rate * 0.9)
                    
                    print(f"Current Learning Rate: {self.learning_rate:.4f}")
            
            # Check termination conditions
            if self.target_fitness is not None and self.best_chromosome.fitness >= self.target_fitness:
                if self.verbose:
                    print(f"\nTarget fitness reached at generation {generation}")
                break
            
            # Early stopping based on validation accuracy stagnation
            if X_val is not None and y_val is not None and generation % eval_interval == 0:
                if len(val_accuracy_history) >= patience and all(val_accuracy_history[-patience:] <= best_val_accuracy):
                    if self.verbose:
                        print(f"\nEarly stopping at generation {generation} due to validation accuracy stagnation")
                    break
            
            # Standard stagnation check
            if stagnation_counter >= self.fitness_stagnation_limit:
                if self.verbose:
                    print(f"\nFitness stagnation detected at generation {generation}")
                break
            
            # Adjust mutation rates based on diversity if using adaptive mutation
            if isinstance(self.mutation_operator, AdaptiveDFAMutation):
                # If diversity is too low, increase mutation rates
                if gene_diversity < 0.05:
                    self.mutation_operator.mutation_rate = min(
                        self.mutation_operator.max_mutation_rate,
                        self.mutation_operator.mutation_rate * 1.2
                    )
                # If diversity is high, decrease mutation rates
                elif gene_diversity > 0.2:
                    self.mutation_operator.mutation_rate = max(
                        self.mutation_operator.min_mutation_rate,
                        self.mutation_operator.mutation_rate * 0.8
                    )
        
        # Close progress bar
        progress_bar.close()
        
        # Final evaluation
        if self.verbose:
            print("\nEvolution completed")
            print(f"Best Fitness: {self.best_chromosome.fitness:.4f}")
            
            if X_val is not None and y_val is not None:
                val_accuracy = self._evaluate_accuracy(self.best_chromosome, X_val, y_val)
                print(f"Final Validation Accuracy: {val_accuracy:.2%}")
            
            if X_test is not None and y_test is not None:
                test_accuracy = self._evaluate_accuracy(self.best_chromosome, X_test, y_test)
                print(f"Final Test Accuracy: {test_accuracy:.2%}")
        
        return self.best_chromosome
    
    def _evaluate_accuracy(self, chromosome: DFAChromosome, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the accuracy of a chromosome on a dataset with optimized performance.
        
        Args:
            chromosome: The chromosome to evaluate.
            X: The input data.
            y: The target labels.
            
        Returns:
            The accuracy of the chromosome on the dataset.
        """
        from tqdm import tqdm
        import numpy as np
        
        # Convert chromosome to DFA
        dfa = chromosome.to_dfa()
        
        # Use a small sample for quick evaluation during evolution
        total = len(X)
        if total > 100:
            # Sample indices using stratified sampling to ensure all classes are represented
            sample_indices = []
            for digit in range(10):
                digit_indices = np.where(y == digit)[0]
                if len(digit_indices) > 0:
                    # Take up to 10 samples per digit
                    digit_samples = np.random.choice(digit_indices, min(10, len(digit_indices)), replace=False)
                    sample_indices.extend(digit_samples)
            
            # Shuffle the indices
            np.random.shuffle(sample_indices)
            
            # Limit total samples
            sample_indices = sample_indices[:100]
            
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
        else:
            X_sample = X
            y_sample = y
        
        # Process in batches for better performance
        batch_size = 10
        correct = 0
        
        for i in range(0, len(X_sample), batch_size):
            batch_end = min(i + batch_size, len(X_sample))
            X_batch = X_sample[i:batch_end]
            y_batch = y_sample[i:batch_end]
            
            for j in range(len(X_batch)):
                # Classify image
                prediction, _ = dfa.classify(X_batch[j])
                
                # Check if prediction is correct
                if prediction == y_batch[j]:
                    correct += 1
        
        # Calculate accuracy
        accuracy = correct / len(X_sample)
        
        return accuracy

    def visualize_feature_importance(self, output_path: str = None) -> None:
        """
        Visualize feature importance maps for each digit.
        
        Args:
            output_path: Path to save the visualization to. If None, the visualization
                        will be displayed but not saved.
        """
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        # Plot feature importance for each digit
        for digit in range(10):
            # Get feature importance map
            importance_map = self.digit_feature_importance[digit]
            
            # Plot
            im = axes[digit].imshow(importance_map, cmap='viridis')
            axes[digit].set_title(f'Digit {digit}')
            axes[digit].axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Feature Importance')
        
        # Add title
        plt.suptitle('Feature Importance Maps for Each Digit', fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()


class DFARandomInitializer(PopulationInitializer):
    """
    Random initializer for DFA chromosomes.
    
    This initializer creates a population of DFA chromosomes with random genes.
    """
    
    def __init__(
        self,
        population_size: int,
        n_states: int = 20,
        n_symbols: int = 256,
        chunk_size: Optional[Tuple[int, int]] = None,
        chunk_stride: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize the DFA random initializer.
        
        Args:
            population_size: The size of the population to create.
            n_states: The number of states in the DFA.
            n_symbols: The number of symbols in the alphabet.
            chunk_size: The size of chunks to process.
            chunk_stride: The stride for chunk processing.
        """
        super().__init__(DFAChromosome, population_size)
        
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
    
    def _create_chromosomes(self, *args, **kwargs) -> List[DFAChromosome]:
        """
        Create a list of DFA chromosomes with random genes.
        
        Returns:
            A list of DFA chromosomes with random genes.
        """
        chromosomes = []
        
        # Calculate gene size
        transition_size = self.n_states * self.n_symbols
        state_types_size = self.n_states
        final_mapping_size = 10 * self.n_states
        gene_size = transition_size + state_types_size + final_mapping_size
        
        for _ in range(self.population_size):
            # Create random genes
            genes = np.random.random(gene_size)
            
            # Create chromosome
            chromosome = DFAChromosome(
                genes=genes,
                n_states=self.n_states,
                n_symbols=self.n_symbols,
                chunk_size=self.chunk_size,
                chunk_stride=self.chunk_stride
            )
            
            chromosomes.append(chromosome)
        
        return chromosomes


class DFAHeuristicInitializer(PopulationInitializer):
    """
    Heuristic initializer for DFA chromosomes.
    
    This initializer creates a population of DFA chromosomes using heuristics
    to guide the initialization process, potentially creating better starting points
    for the genetic algorithm.
    """
    
    def __init__(
        self,
        population_size: int,
        n_states: int = 20,
        n_symbols: int = 256,
        chunk_size: Optional[Tuple[int, int]] = None,
        chunk_stride: Optional[Tuple[int, int]] = None,
        random_proportion: float = 0.5
    ):
        """
        Initialize the DFA heuristic initializer.
        
        Args:
            population_size: The size of the population to create.
            n_states: The number of states in the DFA.
            n_symbols: The number of symbols in the alphabet.
            chunk_size: The size of chunks to process.
            chunk_stride: The stride for chunk processing.
            random_proportion: Proportion of the population to initialize randomly.
        """
        super().__init__(DFAChromosome, population_size)
        
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.random_proportion = random_proportion
    
    def _create_chromosomes(self, *args, **kwargs) -> List[DFAChromosome]:
        """
        Create a list of DFA chromosomes using heuristics.
        
        Returns:
            A list of DFA chromosomes.
        """
        chromosomes = []
        
        # Calculate gene size
        transition_size = self.n_states * self.n_symbols
        state_types_size = self.n_states
        final_mapping_size = 10 * self.n_states
        gene_size = transition_size + state_types_size + final_mapping_size
        
        # Calculate number of random chromosomes
        n_random = int(self.population_size * self.random_proportion)
        n_heuristic = self.population_size - n_random
        
        # Create random chromosomes
        for _ in range(n_random):
            # Create random genes
            genes = np.random.random(gene_size)
            
            # Create chromosome
            chromosome = DFAChromosome(
                genes=genes,
                n_states=self.n_states,
                n_symbols=self.n_symbols,
                chunk_size=self.chunk_size,
                chunk_stride=self.chunk_stride
            )
            
            chromosomes.append(chromosome)
        
        # Create heuristic chromosomes
        for _ in range(n_heuristic):
            # Create genes with heuristic initialization
            genes = self._create_heuristic_genes()
            
            # Create chromosome
            chromosome = DFAChromosome(
                genes=genes,
                n_states=self.n_states,
                n_symbols=self.n_symbols,
                chunk_size=self.chunk_size,
                chunk_stride=self.chunk_stride
            )
            
            chromosomes.append(chromosome)
        
        return chromosomes
    
    def _create_heuristic_genes(self) -> np.ndarray:
        """
        Create genes using heuristics.
        
        Returns:
            A numpy array of genes.
        """
        # Calculate gene sections
        transition_size = self.n_states * self.n_symbols
        state_types_size = self.n_states
        final_mapping_size = 10 * self.n_states
        gene_size = transition_size + state_types_size + final_mapping_size
        
        # Initialize genes
        genes = np.zeros(gene_size)
        
        # Create transition matrix with heuristic initialization
        transition_matrix = np.zeros((self.n_states, self.n_symbols))
        
        # Heuristic 1: Create a linear path through states for low symbols
        # This creates a simple path: state 0 -> state 1 -> state 2 -> ... for symbol 0
        for i in range(self.n_states - 1):
            transition_matrix[i, 0] = (i + 1) / self.n_states  # Normalized to [0, 1]
        
        # Heuristic 2: Create loops for high symbols
        # This creates self-loops for high-value symbols
        for i in range(self.n_states):
            transition_matrix[i, self.n_symbols - 1] = i / self.n_states  # Normalized to [0, 1]
        
        # Heuristic 3: Create random transitions for other symbols
        for i in range(self.n_states):
            for j in range(1, self.n_symbols - 1):
                transition_matrix[i, j] = np.random.random()
        
        # Flatten transition matrix
        genes[:transition_size] = transition_matrix.flatten()
        
        # Create state types with heuristic initialization
        state_types = np.zeros(self.n_states)
        
        # Heuristic 4: First state is always initial
        state_types[0] = STATE_INITIAL / 3.0  # Normalized to [0, 1]
        
        # Heuristic 5: Last few states are final
        for i in range(max(1, self.n_states - 3), self.n_states):
            state_types[i] = STATE_FINAL / 3.0  # Normalized to [0, 1]
        
        # Heuristic 6: Middle states are transition states
        for i in range(1, max(1, self.n_states - 3)):
            state_types[i] = STATE_TRANSITION / 3.0  # Normalized to [0, 1]
        
        # Add state types to genes
        genes[transition_size:transition_size + state_types_size] = state_types
        
        # Create final state mapping with heuristic initialization
        final_mapping = np.zeros((10, self.n_states))
        
        # Heuristic 7: Assign each digit to different final states with high probability
        for digit in range(10):
            # Choose a final state for this digit
            if digit < min(10, self.n_states):
                final_state = self.n_states - digit - 1
                final_mapping[digit, final_state] = 0.9  # High probability
            else:
                # If we have fewer states than digits, assign randomly
                final_state = np.random.randint(max(1, self.n_states - 3), self.n_states)
                final_mapping[digit, final_state] = 0.7  # Medium-high probability
        
        # Add some noise to final mapping
        final_mapping += np.random.random((10, self.n_states)) * 0.1
        
        # Normalize final mapping
        for digit in range(10):
            final_mapping[digit] = final_mapping[digit] / np.sum(final_mapping[digit])
        
        # Flatten final mapping
        genes[transition_size + state_types_size:] = final_mapping.flatten()
        
        return genes


class DFACrossover(CrossoverOperator):
    """
    Crossover operator for DFA chromosomes.
    
    This operator performs crossover between two DFA chromosomes.
    """
    
    def __init__(self, crossover_type: str = 'uniform'):
        """
        Initialize the DFA crossover operator.
        
        Args:
            crossover_type: The type of crossover to perform ('uniform', 'single_point', 'two_point', 'section').
        """
        self.crossover_type = crossover_type
    
    def crossover(self, parent1: DFAChromosome, parent2: DFAChromosome) -> Tuple[DFAChromosome, DFAChromosome]:
        """
        Perform crossover between two DFA chromosomes.
        
        Args:
            parent1: The first parent chromosome.
            parent2: The second parent chromosome.
            
        Returns:
            A tuple containing two offspring chromosomes.
        """
        # Check if parents have the same parameters
        if (parent1.n_states != parent2.n_states or
            parent1.n_symbols != parent2.n_symbols or
            parent1.chunk_size != parent2.chunk_size or
            parent1.chunk_stride != parent2.chunk_stride):
            raise ValueError("Parents must have the same parameters")
        
        # Get parent genes
        parent1_genes = parent1.genes
        parent2_genes = parent2.genes
        
        # Calculate gene sections
        n_states = parent1.n_states
        n_symbols = parent1.n_symbols
        
        transition_size = n_states * n_symbols
        state_types_size = n_states
        final_mapping_size = 10 * n_states
        
        # Perform crossover based on type
        if self.crossover_type == 'uniform':
            # Create crossover mask
            mask = np.random.random(parent1_genes.shape) < 0.5
            
            # Create offspring genes
            offspring1_genes = np.where(mask, parent1_genes, parent2_genes)
            offspring2_genes = np.where(mask, parent2_genes, parent1_genes)
            
        elif self.crossover_type == 'single_point':
            # Choose crossover point
            crossover_point = np.random.randint(1, len(parent1_genes))
            
            # Create offspring genes
            offspring1_genes = np.concatenate([parent1_genes[:crossover_point], parent2_genes[crossover_point:]])
            offspring2_genes = np.concatenate([parent2_genes[:crossover_point], parent1_genes[crossover_point:]])

        elif self.crossover_type == 'two_point':
            # Choose two crossover points
            crossover_points = sorted(np.random.randint(1, len(parent1_genes) - 1, size=2))
            
            # Create offspring genes
            offspring1_genes = np.concatenate([
                parent1_genes[:crossover_points[0]],
                parent2_genes[crossover_points[0]:crossover_points[1]],
                parent1_genes[crossover_points[1]:]
            ])
            offspring2_genes = np.concatenate([
                parent2_genes[:crossover_points[0]],
                parent1_genes[crossover_points[0]:crossover_points[1]],
                parent2_genes[crossover_points[1]:]
            ])
        
        elif self.crossover_type == 'section':
            # Choose a section of genes to swap
            section_size = np.random.randint(1, len(parent1_genes) // 2)
            crossover_point = np.random.randint(0, len(parent1_genes) - section_size)

            # Create offspring genes
            offspring1_genes = parent1_genes[:crossover_point] + parent2_genes[crossover_point:crossover_point+section_size] + parent1_genes[crossover_point+section_size:]
            offspring2_genes = parent2_genes[:crossover_point] + parent1_genes[crossover_point:crossover_point+section_size] + parent2_genes[crossover_point+section_size:]
        
        else:
            raise ValueError("Invalid crossover type")
        
        # Create offspring chromosomes
        offspring1 = DFAChromosome(
            genes=offspring1_genes,
            n_states=n_states,
            n_symbols=n_symbols,
            chunk_size=parent1.chunk_size,
            chunk_stride=parent1.chunk_stride
        )
        
        offspring2 = DFAChromosome(
            genes=offspring2_genes,
            n_states=n_states,
            n_symbols=n_symbols,
            chunk_size=parent1.chunk_size,
            chunk_stride=parent1.chunk_stride
        )
        
        return offspring1, offspring2


class DFAMutation(MutationOperator):
    """
    Mutation operator for DFA chromosomes.
    
    This operator performs mutation on a DFA chromosome.
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.01,
        transition_mutation_rate: float = 0.01,
        state_type_mutation_rate: float = 0.05,
        final_mapping_mutation_rate: float = 0.05
    ):
        """
        Initialize the DFA mutation operator.
        
        Args:
            mutation_rate: The overall probability of mutation.
            transition_mutation_rate: The probability of mutating each transition.
            state_type_mutation_rate: The probability of mutating each state type.
            final_mapping_mutation_rate: The probability of mutating each final state mapping.
        """
        self.mutation_rate = mutation_rate
        self.transition_mutation_rate = transition_mutation_rate
        self.state_type_mutation_rate = state_type_mutation_rate
        self.final_mapping_mutation_rate = final_mapping_mutation_rate
    
    def mutate(self, chromosome: DFAChromosome) -> DFAChromosome:
        """
        Perform mutation on a DFA chromosome.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome.
        """
        # Check if mutation should be applied
        if random.random() > self.mutation_rate:
            return chromosome.clone()
        
        # Create a copy of the chromosome
        mutated = chromosome.clone()
        
        # Calculate gene sections
        n_states = chromosome.n_states
        n_symbols = chromosome.n_symbols
        
        transition_size = n_states * n_symbols
        state_types_size = n_states
        final_mapping_size = 10 * n_states
        
        # Mutate transition matrix
        transition_mask = np.random.random(transition_size) < self.transition_mutation_rate
        if np.any(transition_mask):
            # Generate random transitions
            random_transitions = np.random.random(transition_size)
            
            # Apply mutations
            mutated.genes[:transition_size][transition_mask] = random_transitions[transition_mask]
        
        # Mutate state types
        state_types_mask = np.random.random(state_types_size) < self.state_type_mutation_rate
        if np.any(state_types_mask):
            # Generate random state types
            random_state_types = np.random.random(state_types_size)
            
            # Apply mutations
            mutated.genes[transition_size:transition_size + state_types_size][state_types_mask] = random_state_types[state_types_mask]
        
        # Mutate final state mapping
        final_mapping_mask = np.random.random(final_mapping_size) < self.final_mapping_mutation_rate
        if np.any(final_mapping_mask):
            # Generate random final state mappings
            random_final_mappings = np.random.random(final_mapping_size)
            
            # Apply mutations
            mutated.genes[transition_size + state_types_size:][final_mapping_mask] = random_final_mappings[final_mapping_mask]
        
        # Ensure first state is initial
        # This is handled in the to_dfa method of DFAChromosome
        
        return mutated


class AdaptiveDFAMutation(DFAMutation):
    """
    Adaptive mutation operator for DFA chromosomes.
    
    This operator adjusts mutation rates based on population diversity and fitness.
    """
    
    def __init__(
        self,
        mutation_rate: float = 0.01,
        transition_mutation_rate: float = 0.01,
        state_type_mutation_rate: float = 0.05,
        final_mapping_mutation_rate: float = 0.05,
        min_mutation_rate: float = 0.005,
        max_mutation_rate: float = 0.3,
        adaptation_rate: float = 0.1
    ):
        """
        Initialize the adaptive DFA mutation operator.
        
        Args:
            mutation_rate: The initial overall probability of mutation.
            transition_mutation_rate: The initial probability of mutating each transition.
            state_type_mutation_rate: The initial probability of mutating each state type.
            final_mapping_mutation_rate: The initial probability of mutating each final state mapping.
            min_mutation_rate: The minimum mutation rate.
            max_mutation_rate: The maximum mutation rate.
            adaptation_rate: The rate at which mutation rates adapt.
        """
        super().__init__(
            mutation_rate=mutation_rate,
            transition_mutation_rate=transition_mutation_rate,
            state_type_mutation_rate=state_type_mutation_rate,
            final_mapping_mutation_rate=final_mapping_mutation_rate
        )
        
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.adaptation_rate = adaptation_rate
        self.generation_count = 0
        self.last_best_fitness = 0
        self.stagnation_count = 0
    
    def update_rates(self, population: List[DFAChromosome]) -> None:
        """
        Update mutation rates based on population diversity and fitness.
        
        Args:
            population: The current population.
        """
        self.generation_count += 1
        
        # Calculate population diversity
        if len(population) > 1:
            # Calculate average gene values for each position
            avg_genes = np.mean([c.genes for c in population], axis=0)
            
            # Calculate gene diversity as standard deviation from average
            gene_diversity = np.mean(np.std([c.genes for c in population], axis=0))
            
            # Calculate fitness diversity
            fitness_values = np.array([c.fitness for c in population])
            fitness_diversity = np.std(fitness_values) if len(fitness_values) > 1 else 0
            
            # Get best fitness
            best_fitness = max(fitness_values) if len(fitness_values) > 0 else 0
            
            # Check for fitness stagnation
            if best_fitness <= self.last_best_fitness:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
                self.last_best_fitness = best_fitness
            
            # Adjust mutation rates based on diversity and stagnation
            diversity_factor = 1.0 - min(1.0, gene_diversity * 10)  # Lower diversity -> higher factor
            stagnation_factor = min(1.0, self.stagnation_count / 5)  # More stagnation -> higher factor
            
            # Calculate new mutation rates
            new_mutation_rate = self.mutation_rate * (1.0 + self.adaptation_rate * (diversity_factor + stagnation_factor))
            new_transition_rate = self.transition_mutation_rate * (1.0 + self.adaptation_rate * (diversity_factor + stagnation_factor))
            new_state_type_rate = self.state_type_mutation_rate * (1.0 + self.adaptation_rate * (diversity_factor + stagnation_factor))
            new_final_mapping_rate = self.final_mapping_mutation_rate * (1.0 + self.adaptation_rate * (diversity_factor + stagnation_factor))
            
            # Apply bounds
            self.mutation_rate = np.clip(new_mutation_rate, self.min_mutation_rate, self.max_mutation_rate)
            self.transition_mutation_rate = np.clip(new_transition_rate, self.min_mutation_rate, self.max_mutation_rate)
            self.state_type_mutation_rate = np.clip(new_state_type_rate, self.min_mutation_rate, self.max_mutation_rate)
            self.final_mapping_mutation_rate = np.clip(new_final_mapping_rate, self.min_mutation_rate, self.max_mutation_rate)
    
    def mutate(self, chromosome: DFAChromosome) -> DFAChromosome:
        """
        Perform adaptive mutation on a DFA chromosome.
        
        Args:
            chromosome: The chromosome to mutate.
            
        Returns:
            The mutated chromosome.
        """
        # Apply standard mutation with current rates
        return super().mutate(chromosome)


class IntelligentDFACrossover(DFACrossover):
    """
    Intelligent crossover operator for DFA chromosomes.
    
    This operator performs more sophisticated crossover between two DFA chromosomes,
    preserving successful state transitions and patterns.
    """
    
    def __init__(self, crossover_type: str = 'intelligent'):
        """
        Initialize the intelligent DFA crossover operator.
        
        Args:
            crossover_type: The type of crossover to perform ('intelligent', 'state_preserving', 'transition_aware').
        """
        super().__init__(crossover_type='section')  # Default to section crossover as fallback
        self.crossover_type = crossover_type
    
    def crossover(self, parent1: DFAChromosome, parent2: DFAChromosome) -> Tuple[DFAChromosome, DFAChromosome]:
        """
        Perform intelligent crossover between two DFA chromosomes.
        
        Args:
            parent1: The first parent chromosome.
            parent2: The second parent chromosome.
            
        Returns:
            A tuple containing two offspring chromosomes.
        """
        # Check if parents have the same parameters
        if (parent1.n_states != parent2.n_states or
            parent1.n_symbols != parent2.n_symbols or
            parent1.chunk_size != parent2.chunk_size or
            parent1.chunk_stride != parent2.chunk_stride):
            raise ValueError("Parents must have the same parameters")
        
        # Get parent genes
        parent1_genes = parent1.genes
        parent2_genes = parent2.genes
        
        # Calculate gene sections
        n_states = parent1.n_states
        n_symbols = parent1.n_symbols
        
        transition_size = n_states * n_symbols
        state_types_size = n_states
        final_mapping_size = 10 * n_states
        
        # Convert parents to DFAs to analyze their structure
        dfa1 = parent1.to_dfa()
        dfa2 = parent2.to_dfa()
        
        # Determine which parent has higher fitness
        better_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
        worse_parent = parent2 if parent1.fitness >= parent2.fitness else parent1
        
        better_dfa = dfa1 if parent1.fitness >= parent2.fitness else dfa2
        worse_dfa = dfa2 if parent1.fitness >= parent2.fitness else dfa1
        
        if self.crossover_type == 'intelligent':
            # Create offspring genes based on parent fitness
            # Take more genes from the better parent
            fitness_ratio = max(0.5, min(0.9, better_parent.fitness / (better_parent.fitness + worse_parent.fitness + 1e-10)))
            
            # Create crossover mask weighted by fitness
            mask = np.random.random(parent1_genes.shape) < fitness_ratio
            
            # Create offspring genes
            offspring1_genes = np.where(mask, better_parent.genes, worse_parent.genes)
            
            # For the second offspring, use more randomization to maintain diversity
            diversity_mask = np.random.random(parent1_genes.shape) < 0.7
            offspring2_genes = np.where(diversity_mask, worse_parent.genes, better_parent.genes)
            
        elif self.crossover_type == 'state_preserving':
            # Identify final states in both DFAs
            final_states1 = set(np.where(dfa1.state_types == 1)[0])  # STATE_FINAL = 1
            final_states2 = set(np.where(dfa2.state_types == 2)[0])
            
            # Create offspring by preserving final states from both parents
            offspring1_state_types = parent1.genes[transition_size:transition_size + state_types_size].copy()
            offspring2_state_types = parent2.genes[transition_size:transition_size + state_types_size].copy()
            
            # Preserve transitions to final states from better parent
            offspring1_transition = parent1.genes[:transition_size].reshape(n_states, n_symbols).copy()
            offspring2_transition = parent2.genes[:transition_size].reshape(n_states, n_symbols).copy()
            
            # Mix final state mappings
            offspring1_final_mapping = np.zeros(final_mapping_size)
            offspring2_final_mapping = np.zeros(final_mapping_size)
            
            # For each digit (0-9), take the mapping from the parent with better recognition
            for digit in range(10):
                parent1_mapping = parent1.genes[transition_size + state_types_size + digit * n_states:(transition_size + state_types_size + (digit + 1) * n_states)]
                parent2_mapping = parent2.genes[transition_size + state_types_size + digit * n_states:(transition_size + state_types_size + (digit + 1) * n_states)]
                
                # Choose mapping based on max value (higher confidence)
                if np.max(parent1_mapping) > np.max(parent2_mapping):
                    offspring1_final_mapping[digit * n_states:(digit + 1) * n_states] = parent1_mapping
                    offspring2_final_mapping[digit * n_states:(digit + 1) * n_states] = parent1_mapping
                else:
                    offspring1_final_mapping[digit * n_states:(digit + 1) * n_states] = parent2_mapping
                    offspring2_final_mapping[digit * n_states:(digit + 1) * n_states] = parent2_mapping
            
            # Combine sections
            offspring1_genes = np.concatenate([
                offspring1_transition.flatten(),
                offspring1_state_types,
                offspring1_final_mapping
            ])
            
            offspring2_genes = np.concatenate([
                offspring2_transition.flatten(),
                offspring2_state_types,
                offspring2_final_mapping
            ])
            
        elif self.crossover_type == 'transition_aware':
            # Analyze transition patterns in both DFAs
            # Count how often each state is reached in better_dfa
            state_usage_better = np.zeros(n_states)
            for i in range(n_states):
                for j in range(n_symbols):
                    next_state = better_dfa.transition_matrix[i, j]
                    state_usage_better[next_state] += 1
            
            # Normalize state usage
            state_importance = state_usage_better / (np.sum(state_usage_better) + 1e-10)
            
            # Create transition matrices for offspring
            offspring1_transition = better_parent.genes[:transition_size].reshape(n_states, n_symbols).copy()
            offspring2_transition = worse_parent.genes[:transition_size].reshape(n_states, n_symbols).copy()
            
            # For important states in better parent, preserve their transitions in offspring1
            # For less important states, take transitions from worse parent
            for i in range(n_states):
                if state_importance[i] < 0.05:  # Low importance state
                    # Take transitions from worse parent with some probability
                    if random.random() < 0.7:
                        offspring1_transition[i] = worse_dfa.transition_matrix[i]
            
            # For offspring2, do the opposite to maintain diversity
            for i in range(n_states):
                if state_importance[i] > 0.1:  # High importance state
                    # Take transitions from better parent with some probability
                    if random.random() < 0.6:
                        offspring2_transition[i] = better_dfa.transition_matrix[i]
            
            # Mix state types and final mappings using section crossover
            if random.random() < 0.5:
                offspring1_state_types = better_parent.genes[transition_size:transition_size + state_types_size]
                offspring2_state_types = worse_parent.genes[transition_size:transition_size + state_types_size]
            else:
                offspring1_state_types = worse_parent.genes[transition_size:transition_size + state_types_size]
                offspring2_state_types = better_parent.genes[transition_size:transition_size + state_types_size]
            
            if random.random() < 0.5:
                offspring1_final_mapping = better_parent.genes[transition_size + state_types_size:]
                offspring2_final_mapping = worse_parent.genes[transition_size + state_types_size:]
            else:
                offspring1_final_mapping = worse_parent.genes[transition_size + state_types_size:]
                offspring2_final_mapping = better_parent.genes[transition_size + state_types_size:]
            
            # Combine sections
            offspring1_genes = np.concatenate([
                offspring1_transition.flatten(),
                offspring1_state_types,
                offspring1_final_mapping
            ])
            
            offspring2_genes = np.concatenate([
                offspring2_transition.flatten(),
                offspring2_state_types,
                offspring2_final_mapping
            ])
        
        else:
            # Fall back to standard section crossover
            return super().crossover(parent1, parent2)
        
        # Create offspring chromosomes
        offspring1 = DFAChromosome(
            genes=offspring1_genes,
            n_states=n_states,
            n_symbols=n_symbols,
            chunk_size=parent1.chunk_size,
            chunk_stride=parent1.chunk_stride
        )
        
        offspring2 = DFAChromosome(
            genes=offspring2_genes,
            n_states=n_states,
            n_symbols=n_symbols,
            chunk_size=parent1.chunk_size,
            chunk_stride=parent1.chunk_stride
        )
        
        return offspring1, offspring2
