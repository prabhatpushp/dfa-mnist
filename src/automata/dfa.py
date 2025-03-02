import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union, Set
import copy
from src.automata.base import DeterministicFiniteAutomaton
from src.genetic_algorithm.base import Chromosome


# Define state types as constants
STATE_INITIAL = 0
STATE_FINAL = 1
STATE_TRANSITION = 2
STATE_DEAD = 3


class DFA(DeterministicFiniteAutomaton):
    """
    Implementation of a Deterministic Finite Automaton (DFA) for MNIST.
    
    This DFA uses a transition matrix to efficiently process input sequences.
    It supports both full image processing and chunk-based processing.
    """
    
    def __init__(
        self,
        n_states: int,
        n_symbols: int = 256,
        chunk_size: Optional[Tuple[int, int]] = None,
        chunk_stride: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize the DFA.
        
        Args:
            n_states: The number of states in the DFA.
            n_symbols: The number of symbols in the alphabet (default: 256 for pixel values).
            chunk_size: The size of chunks to process (if None, process full image).
            chunk_stride: The stride for chunk processing (if None, use chunk_size).
        """
        # Basic DFA parameters
        self.n_states = n_states
        self.n_symbols = n_symbols
        
        # Initialize transition matrix
        # Shape: (n_states, n_symbols)
        self.transition_matrix = np.zeros((n_states, n_symbols), dtype=np.int32)
        
        # Initialize state types
        # 0: Initial state, 1: Final state, 2: Transition state, 3: Dead state
        self.state_types = np.zeros(n_states, dtype=np.int32)
        self.state_types[0] = STATE_INITIAL  # First state is initial
        
        # Initialize final state mapping
        # Shape: (10, n_states) for 10 digits
        self.final_state_mapping = np.zeros((10, n_states), dtype=np.float32)
        
        # Set current state to initial state
        self.current_state = 0
        
        # Initialize state history
        self.state_history = []
        
        # Set chunk parameters
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride if chunk_stride is not None else chunk_size
        self.use_chunks = chunk_size is not None
        
        # Add classification cache for performance
        self._classification_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = 1000  # Limit cache size to prevent memory issues
    
    def reset(self) -> None:
        """
        Reset the DFA to its initial state.
        """
        self.current_state = 0
        self.state_history = []
    
    def get_current_state(self) -> int:
        """
        Get the current state of the DFA.
        
        Returns:
            The current state of the DFA.
        """
        return self.current_state
    
    def is_in_final_state(self) -> bool:
        """
        Check if the DFA is in a final state.
        
        Returns:
            True if the DFA is in a final state, False otherwise.
        """
        return self.state_types[self.current_state] == STATE_FINAL
    
    def get_num_states(self) -> int:
        """
        Get the number of states in the DFA.
        
        Returns:
            The number of states in the DFA.
        """
        return self.n_states
    
    def get_alphabet(self) -> Set[int]:
        """
        Get the alphabet of the DFA.
        
        Returns:
            The set of symbols in the alphabet.
        """
        return set(range(self.n_symbols))
    
    def get_transition_function(self) -> Dict:
        """
        Get the transition function of the DFA.
        
        Returns:
            The transition function as a dictionary.
        """
        transition_dict = {}
        for state in range(self.n_states):
            transition_dict[state] = {}
            for symbol in range(self.n_symbols):
                next_state = self.transition_matrix[state, symbol]
                if next_state != 0:  # Ignore transitions to state 0 (default)
                    transition_dict[state][symbol] = next_state
        
        return transition_dict
    
    def get_initial_state(self) -> int:
        """
        Get the initial state of the DFA.
        
        Returns:
            The initial state of the DFA.
        """
        return 0  # First state is always the initial state
    
    def get_final_states(self) -> Set[int]:
        """
        Get the final states of the DFA.
        
        Returns:
            The set of final states of the DFA.
        """
        return set(np.where(self.state_types == STATE_FINAL)[0])
    
    def transition(self, input_symbol: int) -> None:
        """
        Transition to the next state based on the input symbol.
        
        Args:
            input_symbol: The input symbol to process.
        """
        # Ensure input symbol is within range
        if input_symbol < 0 or input_symbol >= self.n_symbols:
            # If out of range, stay in the same state
            return
        
        # Record current state for history
        self.state_history.append(self.current_state)
        
        # Transition to the next state
        self.current_state = self.transition_matrix[self.current_state, input_symbol]
    
    def process_input(self, input_sequence: np.ndarray) -> bool:
        """
        Process an input sequence and determine if it is accepted.
        
        Args:
            input_sequence: The input sequence to process (MNIST image).
            
        Returns:
            True if the input sequence is accepted, False otherwise.
        """
        # Reset the DFA
        self.reset()
        
        # Process input based on chunk configuration
        if self.use_chunks:
            return self._process_chunks(input_sequence)
        else:
            return self._process_full_image(input_sequence)
    
    def _process_full_image(self, image: np.ndarray) -> bool:
        """
        Process a full image using vectorized operations for better performance.
        
        Args:
            image: The image to process.
            
        Returns:
            True if the image is accepted, False otherwise.
        """
        # Flatten the image and convert to symbols in one step
        flat_image = image.flatten()
        symbols = np.minimum((flat_image * (self.n_symbols - 1)).astype(np.int32), self.n_symbols - 1)
        
        # Process in batches for better performance
        batch_size = 64  # Process 64 pixels at a time
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            
            for symbol in batch:
                # Record current state for history if needed
                if len(self.state_history) < 1000:  # Limit history size for performance
                    self.state_history.append(self.current_state)
                
                # Transition to the next state
                self.current_state = self.transition_matrix[self.current_state, symbol]
        
        # Check if the final state is an accepting state
        return self.is_in_final_state()
    
    def _process_chunks(self, image: np.ndarray) -> bool:
        """
        Process an image in chunks using optimized vectorized operations.
        
        Args:
            image: The image to process.
            
        Returns:
            True if the image is accepted, False otherwise.
        """
        # Get image dimensions
        height, width = image.shape
        
        # Get chunk parameters
        chunk_height, chunk_width = self.chunk_size
        stride_height, stride_width = self.chunk_stride
        
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
            symbol = min(int(avg_value * (self.n_symbols - 1)), self.n_symbols - 1)
            
            # Transition to the next state directly (bypass transition method for speed)
            if len(self.state_history) < 1000:  # Limit history size for performance
                self.state_history.append(self.current_state)
            
            self.current_state = self.transition_matrix[self.current_state, symbol]
        
        # Check if the final state is an accepting state
        return self.is_in_final_state()
    
    def classify(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Classify an MNIST image with caching for better performance.
        
        Args:
            image: The MNIST image to classify.
            
        Returns:
            A tuple containing the predicted digit and the confidence scores for each digit.
        """
        # Create a cache key from the image
        # Use a hash of the flattened image as the key
        image_flat = image.flatten()
        cache_key = hash(image_flat.tobytes())
        
        # Check if the image is in the cache
        if cache_key in self._classification_cache:
            self._cache_hits += 1
            return self._classification_cache[cache_key]
        
        self._cache_misses += 1
        
        # Process the image
        self.process_input(image)
        
        # Get the final state
        final_state = self.current_state
        
        # Get the confidence scores for each digit
        confidence_scores = self.final_state_mapping[:, final_state]
        
        # Get the predicted digit
        predicted_digit = np.argmax(confidence_scores)
        
        result = (predicted_digit, confidence_scores)
        
        # Add to cache if not too large
        if len(self._classification_cache) < self._max_cache_size:
            self._classification_cache[cache_key] = result
        
        return result
    
    def clone(self) -> 'DFA':
        """
        Create a deep copy of the DFA.
        
        Returns:
            A new DFA with the same parameters and transition matrix.
        """
        # Create a new DFA with the same parameters
        dfa_clone = DFA(
            n_states=self.n_states,
            n_symbols=self.n_symbols,
            chunk_size=self.chunk_size,
            chunk_stride=self.chunk_stride
        )
        
        # Copy the transition matrix and state types
        dfa_clone.transition_matrix = self.transition_matrix.copy()
        dfa_clone.state_types = self.state_types.copy()
        dfa_clone.final_state_mapping = self.final_state_mapping.copy()
        
        return dfa_clone
    
    def optimize(self) -> 'DFA':
        """
        Optimize the DFA by removing unreachable states.
        
        Returns:
            An optimized DFA.
        """
        # Create a new DFA
        optimized_dfa = self.clone()
        
        # Find reachable states
        reachable_states = set([0])  # Initial state is always reachable
        frontier = [0]
        
        while frontier:
            state = frontier.pop(0)
            for symbol in range(self.n_symbols):
                next_state = self.transition_matrix[state, symbol]
                if next_state not in reachable_states:
                    reachable_states.add(next_state)
                    frontier.append(next_state)
        
        # Mark unreachable states as dead states
        for state in range(self.n_states):
            if state not in reachable_states:
                optimized_dfa.state_types[state] = STATE_DEAD
                optimized_dfa.transition_matrix[state, :] = 0
        
        return optimized_dfa
    
    def to_dict(self) -> Dict:
        """
        Convert the DFA to a dictionary.
        
        Returns:
            A dictionary representation of the DFA.
        """
        return {
            'n_states': self.n_states,
            'n_symbols': self.n_symbols,
            'transition_matrix': self.transition_matrix.tolist(),
            'state_types': self.state_types.tolist(),
            'final_state_mapping': self.final_state_mapping.tolist(),
            'chunk_size': self.chunk_size,
            'chunk_stride': self.chunk_stride,
            'use_chunks': self.use_chunks
        }
    
    @classmethod
    def from_dict(cls, dfa_dict: Dict) -> 'DFA':
        """
        Create a DFA from a dictionary.
        
        Args:
            dfa_dict: A dictionary representation of a DFA.
            
        Returns:
            A new DFA with the parameters from the dictionary.
        """
        dfa = cls(
            n_states=dfa_dict['n_states'],
            n_symbols=dfa_dict['n_symbols'],
            chunk_size=dfa_dict['chunk_size'],
            chunk_stride=dfa_dict['chunk_stride']
        )
        
        dfa.transition_matrix = np.array(dfa_dict['transition_matrix'])
        dfa.state_types = np.array(dfa_dict['state_types'])
        dfa.final_state_mapping = np.array(dfa_dict['final_state_mapping'])
        dfa.use_chunks = dfa_dict['use_chunks']
        
        return dfa


class DFAChromosome(Chromosome):
    """
    Chromosome representation for a DFA.
    
    This chromosome encodes a DFA for MNIST digit recognition.
    """
    
    def __init__(
        self,
        genes: np.ndarray,
        n_states: int = 20,
        n_symbols: int = 256,
        chunk_size: Optional[Tuple[int, int]] = None,
        chunk_stride: Optional[Tuple[int, int]] = None,
        fitness: float = 0.0
    ):
        """
        Initialize the DFA chromosome.
        
        Args:
            genes: The genes of the chromosome.
            n_states: The number of states in the DFA.
            n_symbols: The number of symbols in the alphabet.
            chunk_size: The size of chunks to process.
            chunk_stride: The stride for chunk processing.
            fitness: The fitness value of the chromosome.
        """
        super().__init__(genes, fitness)
        
        self.n_states = n_states
        self.n_symbols = n_symbols
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        
        # Cache for the DFA
        self._dfa = None
    
    def to_dfa(self) -> DFA:
        """
        Convert the chromosome to a DFA.
        
        Returns:
            A DFA represented by the chromosome.
        """
        if self._dfa is not None:
            return self._dfa
        
        # Create a new DFA
        dfa = DFA(
            n_states=self.n_states,
            n_symbols=self.n_symbols,
            chunk_size=self.chunk_size,
            chunk_stride=self.chunk_stride
        )
        
        # Extract transition matrix from genes
        # The genes are structured as follows:
        # - First n_states * n_symbols elements: Transition matrix (flattened)
        # - Next n_states elements: State types
        # - Next 10 * n_states elements: Final state mapping (flattened)
        
        # Calculate sizes
        transition_size = self.n_states * self.n_symbols
        state_types_size = self.n_states
        final_mapping_size = 10 * self.n_states
        
        # Extract transition matrix
        transition_genes = self.genes[:transition_size]
        transition_matrix = transition_genes.reshape((self.n_states, self.n_symbols))
        
        # Ensure transitions are valid state indices
        transition_matrix = np.clip(
            transition_matrix,
            0,
            self.n_states - 1
        ).astype(np.int32)
        
        # Extract state types
        state_types_genes = self.genes[transition_size:transition_size + state_types_size]
        state_types = np.clip(state_types_genes * 4, 0, 3).astype(np.int32)
        
        # Ensure first state is initial
        state_types[0] = STATE_INITIAL
        
        # Extract final state mapping
        final_mapping_genes = self.genes[
            transition_size + state_types_size:
            transition_size + state_types_size + final_mapping_size
        ]
        final_state_mapping = final_mapping_genes.reshape((10, self.n_states))
        
        # Ensure final state mapping is non-negative
        final_state_mapping = np.maximum(final_state_mapping, 0)
        
        # Set DFA parameters
        dfa.transition_matrix = transition_matrix
        dfa.state_types = state_types
        dfa.final_state_mapping = final_state_mapping
        
        # Cache the DFA
        self._dfa = dfa
        
        return dfa
    
    def calculate_fitness(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the fitness of the chromosome.
        
        Args:
            X: The input data (MNIST images).
            y: The target labels (MNIST digits).
            
        Returns:
            The fitness value of the chromosome.
        """
        # Convert chromosome to DFA
        dfa = self.to_dfa()
        
        # Calculate accuracy and penalize incorrect classifications
        correct = 0
        total = len(X)
        penalty = 0
        
        # Track confidence scores for each digit class
        digit_confidence = {i: [] for i in range(10)}
        
        for i in range(total):
            # Classify the image
            predicted_digit, confidence_scores = dfa.classify(X[i])
            
            # Check if prediction is correct
            if predicted_digit == y[i]:
                correct += 1
                # Reward higher confidence in correct predictions (increased reward)
                reward = confidence_scores[predicted_digit] * 0.05
                correct += reward
                
                # Store confidence for this digit class
                digit_confidence[y[i]].append(confidence_scores[predicted_digit])
            else:
                # Penalize incorrect classifications based on confidence
                # Higher confidence in wrong answer = higher penalty (increased penalty)
                penalty += confidence_scores[predicted_digit] * 0.1
                
                # Additional penalty for high confidence in wrong predictions
                if confidence_scores[predicted_digit] > 0.7:
                    penalty += 0.05
        
        # Calculate accuracy
        accuracy = correct / total
        
        # Calculate class balance reward - reward DFAs that can recognize multiple digits
        recognized_classes = sum(1 for digit, confs in digit_confidence.items() if len(confs) > 0)
        class_balance_reward = 0.02 * recognized_classes
        
        # Add complexity penalty with more sophisticated calculation
        # Penalize DFAs with too many states or transitions, but don't penalize too much if accuracy is high
        active_states = np.sum(dfa.state_types != STATE_DEAD)
        complexity_ratio = active_states / self.n_states
        
        # Adaptive complexity penalty - lower penalty for higher accuracy
        complexity_penalty = 0.05 * complexity_ratio * (1.0 - min(0.7, accuracy))
        
        # Calculate transition efficiency - reward DFAs with efficient state transitions
        unique_transitions = len(np.unique(dfa.transition_matrix))
        transition_efficiency = 0.01 * (unique_transitions / (self.n_states * self.n_symbols))
        
        # Calculate fitness (accuracy - penalties + rewards)
        fitness = accuracy - complexity_penalty - (penalty / total) + class_balance_reward - transition_efficiency
        
        return fitness
    
    def clone(self) -> 'DFAChromosome':
        """
        Create a deep copy of the chromosome.
        
        Returns:
            A new chromosome with the same genes.
        """
        return DFAChromosome(
            genes=self.genes.copy(),
            n_states=self.n_states,
            n_symbols=self.n_symbols,
            chunk_size=self.chunk_size,
            chunk_stride=self.chunk_stride,
            fitness=self.fitness
        ) 