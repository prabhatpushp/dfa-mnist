import os
import numpy as np
from src.data.mnist_loader import MNISTLoader
from src.preprocessing.preprocessor import MNISTPreprocessor
from src.utils.visualization import DataVisualizer
from src.utils.config import Config
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
import time
from typing import Dict, Any, List, Tuple, Optional
from src.automata.dfa import DFA, DFAChromosome, STATE_INITIAL, STATE_FINAL, STATE_TRANSITION, STATE_DEAD
from src.automata.dfa_ga import (
    DFAGeneticAlgorithm,
    DFARandomInitializer,
    DFACrossover,
    DFAMutation,
    DFAHeuristicInitializer,
    IntelligentDFACrossover,
    AdaptiveDFAMutation
)
from src.genetic_algorithm.selection import (
    TournamentSelection,
    RouletteWheelSelection,
    RankSelection,
    ElitismSelection
)
from src.genetic_algorithm.metrics import GeneticAlgorithmMetrics
from src.genetic_algorithm.logger import GeneticAlgorithmLogger
from src.genetic_algorithm.config import GeneticAlgorithmConfig, ConfigBuilder

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run DFA-based genetic algorithm on MNIST dataset')
    
    # Data parameters
    parser.add_argument('--limit_samples', type=int, default=1000, help='Limit number of samples (for debugging)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--validation_size', type=float, default=0.2, help='Validation set size')
    parser.add_argument('--binarize', action='store_true', help='Binarize images')
    parser.add_argument('--binarize_threshold', type=float, default=0.5, help='Threshold for binarization')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    
    # DFA parameters
    parser.add_argument('--n_states', type=int, default=10, help='Number of states in DFA')
    parser.add_argument('--n_symbols', type=int, default=10, help='Number of symbols in alphabet')
    parser.add_argument('--use_chunks', action='store_true', help='Use image chunks instead of full images')
    parser.add_argument('--chunk_size', type=int, nargs=2, default=[7, 7], help='Chunk size (height, width)')
    parser.add_argument('--chunk_stride', type=int, nargs=2, default=[7, 7], help='Chunk stride (height, width)')
    
    # Genetic algorithm parameters
    parser.add_argument('--population_size', type=int, default=50, help='Population size')
    parser.add_argument('--n_generations', type=int, default=20, help='Number of generations')
    parser.add_argument('--crossover_rate', type=float, default=0.8, help='Crossover rate')
    parser.add_argument('--mutation_rate', type=float, default=0.2, help='Mutation rate')
    parser.add_argument('--selection_method', type=str, default='tournament', 
                        choices=['tournament', 'roulette', 'rank', 'elitism'],
                        help='Selection method')
    parser.add_argument('--tournament_size', type=int, default=5, help='Tournament size')
    parser.add_argument('--elitism_rate', type=float, default=0.1, help='Elitism rate')
    parser.add_argument('--crossover_type', type=str, default='uniform',
                        choices=['uniform', 'single_point', 'two_point', 'section'],
                        help='Crossover type')
    parser.add_argument('--initializer_type', type=str, default='random',
                        choices=['random', 'heuristic'],
                        help='Population initializer type')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval_interval', type=int, default=5, help='Interval for evaluation on validation/test sets')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--save_best', action='store_true', help='Save best DFA')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    return parser.parse_args()

def setup_output_dir(output_dir: str) -> str:
    """
    Set up output directory.
    
    Args:
        output_dir: Base output directory.
        
    Returns:
        Path to the created output directory.
    """
    # Create timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create output directory
    output_path = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    return output_path

def create_selection_operator(args):
    """
    Create selection operator based on arguments.
    
    Args:
        args: Command line arguments.
        
    Returns:
        Selection operator.
    """
    if args.selection_method == 'tournament':
        return TournamentSelection(tournament_size=args.tournament_size)
    elif args.selection_method == 'roulette':
        return RouletteWheelSelection()
    elif args.selection_method == 'rank':
        return RankSelection()
    elif args.selection_method == 'elitism':
        return ElitismSelection(elitism_rate=args.elitism_rate)
    else:
        raise ValueError(f"Invalid selection method: {args.selection_method}")

def create_initializer(args):
    """
    Create population initializer based on arguments.
    
    Args:
        args: Command line arguments.
        
    Returns:
        Population initializer.
    """
    # Set chunk parameters if using chunks
    chunk_size = tuple(args.chunk_size) if args.use_chunks else None
    chunk_stride = tuple(args.chunk_stride) if args.use_chunks else None
    
    if args.initializer_type == 'random':
        return DFARandomInitializer(
            population_size=args.population_size,
            n_states=args.n_states,
            n_symbols=args.n_symbols,
            chunk_size=chunk_size,
            chunk_stride=chunk_stride
        )
    elif args.initializer_type == 'heuristic':
        return DFAHeuristicInitializer(
            population_size=args.population_size,
            n_states=args.n_states,
            n_symbols=args.n_symbols,
            chunk_size=chunk_size,
            chunk_stride=chunk_stride,
            random_proportion=0.5
        )
    else:
        raise ValueError(f"Invalid initializer type: {args.initializer_type}")

def augment_data(X: np.ndarray, y: np.ndarray, augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment the training data with rotations, shifts, and noise.
    
    Args:
        X: The input data.
        y: The target labels.
        augmentation_factor: How many times to augment each image.
        
    Returns:
        Augmented data and labels.
    """
    from scipy.ndimage import rotate, shift
    
    print(f"Augmenting data (factor: {augmentation_factor})...")
    
    # Get original shape
    if len(X.shape) == 3:
        n_samples, height, width = X.shape
        is_flat = False
    else:
        n_samples, n_pixels = X.shape
        height = width = int(np.sqrt(n_pixels))
        is_flat = True
        # Reshape for augmentation
        X = X.reshape(n_samples, height, width)
    
    # Initialize augmented data
    X_aug = []
    y_aug = []
    
    # Add original data
    X_aug.append(X)
    y_aug.append(y)
    
    # Create augmented versions
    for i in range(augmentation_factor - 1):
        X_rotated = np.zeros_like(X)
        
        for j in range(n_samples):
            # Random rotation
            angle = np.random.uniform(-15, 15)
            X_rotated[j] = rotate(X[j], angle, reshape=False, mode='constant', cval=0)
            
            # Random shift
            shift_x = np.random.randint(-2, 3)
            shift_y = np.random.randint(-2, 3)
            X_rotated[j] = shift(X_rotated[j], [shift_y, shift_x], mode='constant', cval=0)
            
            # Add noise
            noise = np.random.normal(0, 0.05, X_rotated[j].shape)
            X_rotated[j] = np.clip(X_rotated[j] + noise, 0, 1)
        
        X_aug.append(X_rotated)
        y_aug.append(y)
    
    # Concatenate all augmented data
    X_augmented = np.vstack(X_aug)
    y_augmented = np.concatenate(y_aug)
    
    # Reshape back if needed
    if is_flat:
        X_augmented = X_augmented.reshape(X_augmented.shape[0], -1)
    
    print(f"Data augmented: {X.shape[0]} -> {X_augmented.shape[0]} samples")
    
    return X_augmented, y_augmented

def evaluate_dfa(dfa: DFA, X: np.ndarray, y: np.ndarray, chunk_size=None, chunk_stride=None) -> float:
    """
    Evaluate a DFA on a dataset with optimized batch processing.
    
    Args:
        dfa: The DFA to evaluate.
        X: The input data.
        y: The target labels.
        chunk_size: The size of chunks to process (not used, already set in DFA).
        chunk_stride: The stride for chunk processing (not used, already set in DFA).
        
    Returns:
        The accuracy of the DFA on the dataset.
    """
    correct = 0
    total = len(X)
    
    # Process in batches for better performance
    batch_size = 10
    
    # Check if memory monitoring is available
    memory_monitoring = False
    initial_eval_memory = get_memory_usage()
    if initial_eval_memory >= 0:
        memory_monitoring = True
        print(f"Memory usage before evaluation: {initial_eval_memory:.2f} MB")
    
    with tqdm(total=total, desc="Evaluating DFA", leave=False) as pbar:
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            X_batch = X[i:batch_end]
            y_batch = y[i:batch_end]
            
            for j in range(len(X_batch)):
                # Classify image
                prediction, _ = dfa.classify(X_batch[j])
                
                # Check if prediction is correct
                if prediction == y_batch[j]:
                    correct += 1
                
                pbar.update(1)
            
            # Monitor memory usage periodically (every 5 batches)
            if memory_monitoring and i % (batch_size * 5) == 0 and i > 0:
                current_memory = get_memory_usage()
                memory_diff = current_memory - initial_eval_memory
                if memory_diff > 10:  # Only log if significant change (>10MB)
                    print(f"Memory during evaluation ({i}/{total} samples): {current_memory:.2f} MB (+{memory_diff:.2f} MB)")
    
    # Calculate accuracy
    accuracy = correct / total
    
    # Final memory check
    if memory_monitoring:
        final_eval_memory = get_memory_usage()
        memory_diff = final_eval_memory - initial_eval_memory
        print(f"Memory usage after evaluation: {final_eval_memory:.2f} MB (change: {memory_diff:.2f} MB)")
    
    return accuracy

def save_dfa(dfa: DFA, output_path: str, metrics: Dict[str, Any]) -> None:
    """
    Save a DFA to a file.
    
    Args:
        dfa: The DFA to save.
        output_path: The path to save the DFA to.
        metrics: Metrics to save with the DFA.
    """
    # Create DFA data
    dfa_data = {
        'n_states': dfa.n_states,
        'n_symbols': dfa.n_symbols,
        'transition_matrix': dfa.transition_matrix.tolist(),
        'state_types': dfa.state_types.tolist(),
        'final_state_mapping': dfa.final_state_mapping.tolist(),
        'metrics': metrics
    }
    
    # Save DFA data
    with open(output_path, 'w') as f:
        json.dump(dfa_data, f, indent=2)

def visualize_training_progress(metrics: GeneticAlgorithmMetrics, output_path: str) -> None:
    """
    Visualize training progress.
    
    Args:
        metrics: Metrics from the genetic algorithm.
        output_path: Path to save the visualization to.
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot best fitness
    axes[0, 0].plot(metrics.best_fitness_history)
    axes[0, 0].set_title('Best Fitness')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].grid(True)
    
    # Plot average fitness
    axes[0, 1].plot(metrics.avg_fitness_history)
    axes[0, 1].set_title('Average Fitness')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Fitness')
    axes[0, 1].grid(True)
    
    # Plot gene diversity
    axes[1, 0].plot(metrics.gene_diversity_history)
    axes[1, 0].set_title('Gene Diversity')
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].grid(True)
    
    # Plot stagnation count instead of execution time
    axes[1, 1].plot(metrics.stagnation_count_history)
    axes[1, 1].set_title('Stagnation Count')
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Generations without Improvement')
    axes[1, 1].grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def get_memory_usage():
    """
    Get the current memory usage of the process.
    
    Returns:
        Memory usage in MB or -1 if psutil is not available.
    """
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / (1024 * 1024)  # Convert to MB
        
        return memory_usage
    except ImportError:
        print("Warning: psutil package not installed. Memory monitoring disabled.")
        return -1

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Set up output directory
    output_path = setup_output_dir(args.output_dir)
    
    # Save arguments
    with open(os.path.join(output_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Monitor initial memory usage
    initial_memory = get_memory_usage()
    memory_monitoring = initial_memory >= 0
    if memory_monitoring:
        print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Load MNIST dataset
    mnist_loader = MNISTLoader(
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_seed,
        normalize=True,
        binarize=args.binarize,
        binarize_threshold=args.binarize_threshold,
        flatten=False,  # Keep 2D structure for preprocessing
        limit_samples=args.limit_samples
    )
    
    X_train, y_train, X_val, y_val, X_test, y_test = mnist_loader.load_data()
    
    # Apply preprocessing to reduce the number of symbols
    print(f"Original image shape: {X_train.shape}")
    
    # Create preprocessor for reducing the number of symbols
    preprocessor = MNISTPreprocessor(
        n_symbols=args.n_symbols,
        normalize=True,
        noise_reduction=True,  # Enable noise reduction for better performance
        binarize=args.binarize
    )
    
    # Apply preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)
    
    print(f"Preprocessed image shape: {X_train.shape}")
    print(f"Number of symbols reduced to: {args.n_symbols}")
    
    # Apply data augmentation if enabled
    if args.augment:
        X_train, y_train = augment_data(X_train, y_train, augmentation_factor=2)
    
    # Flatten images if not using chunks
    if not args.use_chunks:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Monitor memory after data loading
    if memory_monitoring:
        data_memory = get_memory_usage()
        print(f"Memory usage after data loading: {data_memory:.2f} MB (+ {data_memory - initial_memory:.2f} MB)")
    
    # Set chunk parameters if using chunks
    chunk_size = tuple(args.chunk_size) if args.use_chunks else None
    chunk_stride = tuple(args.chunk_stride) if args.use_chunks else None
    
    # Create selection operator
    selection_operator = create_selection_operator(args)
    
    # Create crossover operator - use the new intelligent crossover
    crossover_operator = IntelligentDFACrossover(crossover_type='intelligent')
    
    # Create mutation operator - use the new adaptive mutation with optimized rates
    mutation_operator = AdaptiveDFAMutation(
        mutation_rate=args.mutation_rate,
        transition_mutation_rate=0.08,  # Increased for better exploration
        state_type_mutation_rate=0.15,  # Increased for better exploration
        final_mapping_mutation_rate=0.15,  # Increased for better exploration
        min_mutation_rate=0.02,  # Increased minimum rate
        max_mutation_rate=0.4,  # Increased maximum rate
        adaptation_rate=0.15  # Increased adaptation rate
    )
    
    # Create population initializer
    initializer = create_initializer(args)
    
    # Create metrics and logger
    metrics = GeneticAlgorithmMetrics()
    logger = GeneticAlgorithmLogger(
        log_file=os.path.join(output_path, 'ga_log.txt'),
        verbose=args.verbose
    )
    
    # Create genetic algorithm with self-learning enabled and optimized parameters
    ga = DFAGeneticAlgorithm(
        n_states=args.n_states,
        n_symbols=args.n_symbols,
        chunk_size=chunk_size,
        chunk_stride=chunk_stride,
        population_size=args.population_size,
        selection_operator=selection_operator,
        crossover_operator=crossover_operator,
        mutation_operator=mutation_operator,
        elite_count=int(args.population_size * 0.15),  # Increased elitism rate
        max_generations=args.n_generations,
        learning_rate=0.2,  # Increased learning rate
        enable_self_learning=True,
        pattern_memory_size=15,  # Increased pattern memory
        # Enhanced learning parameters
        min_learning_rate=0.08,  # Increased minimum learning rate
        max_learning_rate=0.4,  # Increased maximum learning rate
        learning_rate_adaptation=0.15,  # Increased adaptation rate
        feature_importance_threshold=0.15,  # Decreased threshold for more features
        confidence_threshold=0.6,  # Decreased threshold for more patterns
        forget_factor=0.08,  # Increased forget factor
        fitness_stagnation_limit=8,  # Increased stagnation limit
        verbose=args.verbose
    )
    
    # Set the initializer as an attribute after initialization
    ga.initializer = initializer
    
    # Set metrics and logger
    ga.metrics = metrics
    ga.logger = logger
    
    # Monitor memory before GA execution
    pre_ga_memory = -1
    if memory_monitoring:
        pre_ga_memory = get_memory_usage()
        print(f"Memory usage before GA execution: {pre_ga_memory:.2f} MB (+ {pre_ga_memory - data_memory:.2f} MB)")
    
    # Run genetic algorithm
    print(f"Running optimized genetic algorithm for {args.n_generations} generations...")
    print(f"Using intelligent crossover and adaptive mutation with enhanced self-learning...")
    print(f"Number of symbols in alphabet: {args.n_symbols}")
    
    start_time = time.time()
    
    # Pass validation and test data to evolve method for periodic evaluation
    best_chromosome = ga.evolve(
        X_train, y_train, 
        X_val=X_val, y_val=y_val, 
        X_test=X_test, y_test=y_test,
        eval_interval=args.eval_interval
    )
    
    # Calculate execution time
    execution_time = time.time() - start_time
    print(f"GA execution time: {execution_time:.2f} seconds")
    
    # Monitor memory after GA execution
    post_ga_memory = -1
    if memory_monitoring:
        post_ga_memory = get_memory_usage()
        print(f"Memory usage after GA execution: {post_ga_memory:.2f} MB (+ {post_ga_memory - pre_ga_memory:.2f} MB)")
    
    # Convert best chromosome to DFA
    best_dfa = best_chromosome.to_dfa()
    
    # Evaluate best DFA on validation set
    print("Evaluating best DFA on validation set...")
    val_accuracy = evaluate_dfa(best_dfa, X_val, y_val, chunk_size, chunk_stride)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Evaluate best DFA on test set
    print("Evaluating best DFA on test set...")
    test_accuracy = evaluate_dfa(best_dfa, X_test, y_test, chunk_size, chunk_stride)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save best DFA
    if args.save_best:
        print("Saving best DFA...")
        metrics_dict = {
            'train_fitness': best_chromosome.fitness,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'execution_time': execution_time
        }
        
        if memory_monitoring:
            metrics_dict['memory_usage'] = post_ga_memory
            
        save_dfa(
            best_dfa,
            os.path.join(output_path, 'best_dfa.json'),
            metrics_dict
        )
    
    # Visualize training progress
    print("Visualizing training progress...")
    visualize_training_progress(metrics, os.path.join(output_path, 'training_progress.png'))
    
    # Visualize feature importance if self-learning was enabled
    if hasattr(ga, 'enable_self_learning') and ga.enable_self_learning:
        print("Visualizing feature importance maps...")
        ga.visualize_feature_importance(os.path.join(output_path, 'feature_importance.png'))
    
    # Save metrics
    metrics_dict = {
        'best_fitness_history': metrics.best_fitness_history,
        'avg_fitness_history': metrics.avg_fitness_history,
        'gene_diversity_history': metrics.gene_diversity_history,
        'stagnation_count_history': metrics.stagnation_count_history,
        'final_best_fitness': best_chromosome.fitness,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'execution_time': execution_time
    }
    
    if memory_monitoring:
        metrics_dict['initial_memory'] = initial_memory
        metrics_dict['final_memory'] = post_ga_memory
    
    with open(os.path.join(output_path, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Final memory usage
    if memory_monitoring:
        final_memory = get_memory_usage()
        print(f"Final memory usage: {final_memory:.2f} MB (total increase: {final_memory - initial_memory:.2f} MB)")
    
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()
