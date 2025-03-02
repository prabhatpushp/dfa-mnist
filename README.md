# Genetic Algorithm-Based Deterministic Finite Automata for MNIST Digit Recognition 🎉

## Introduction
Hey there! 👋 I'm Prabhat Pushp, and I'm excited to share this project with you! This project implements a novel approach to machine learning by combining genetic algorithms with formal automata theory. The system evolves Deterministic Finite Automata (DFA) to create pattern recognition models for MNIST digit recognition.

Traditional machine learning approaches often rely on complex neural networks that can be difficult to interpret. This project explores an alternative approach using formal automata theory, specifically Deterministic Finite Automata (DFA), to create interpretable models for pattern recognition.

## Features 🌟
- **MNIST Data Processing**: Loading, preprocessing, and formatting MNIST data for use with automata-based learning models.
- **Genetic Algorithm Framework**: A flexible, modular GA framework that can be applied to evolve DFA-based solutions.
- **DFA Implementation**: A DFA class that uses NumPy matrices for efficient state transitions and can process both full images and chunked image data.
- **Visualization Tools**: Tools for visualizing the DFA structure, training progress, and results.

## Installation ⚙️

### Prerequisites
- Python 3.8+
- NumPy
- scikit-learn
- matplotlib
- tqdm

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/prabhatpushp/dfa-mnist
   cd dfa-mnist
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage 📈

### Running the Genetic Algorithm
To run the genetic algorithm with default parameters:
```bash
python main.py
```

### Command Line Arguments
The script supports various command line arguments to customize the behavior:

#### Data Parameters
- `--limit_samples`: Limit number of samples (for debugging)
- `--test_size`: Test set size (default: 0.2)
- `--validation_size`: Validation set size (default: 0.2)
- `--binarize`: Binarize images
- `--binarize_threshold`: Threshold for binarization (default: 0.5)

#### DFA Parameters
- `--n_states`: Number of states in DFA (default: 20)
- `--n_symbols`: Number of symbols in alphabet (default: 256)
- `--use_chunks`: Use image chunks instead of full images
- `--chunk_size`: Chunk size (height, width) (default: [7, 7])
- `--chunk_stride`: Chunk stride (height, width) (default: [7, 7])

#### Genetic Algorithm Parameters
- `--population_size`: Population size (default: 100)
- `--n_generations`: Number of generations (default: 50)
- `--crossover_rate`: Crossover rate (default: 0.8)
- `--mutation_rate`: Mutation rate (default: 0.1)
- `--selection_method`: Selection method (choices: tournament, roulette, rank, elitism) (default: tournament)
- `--tournament_size`: Tournament size (default: 5)
- `--elitism_rate`: Elitism rate (default: 0.1)
- `--crossover_type`: Crossover type (choices: uniform, single_point, two_point, section) (default: uniform)
- `--initializer_type`: Population initializer type (choices: random, heuristic) (default: random)
- `--random_seed`: Random seed (default: 42)

#### Output Parameters
- `--output_dir`: Output directory (default: results)
- `--save_best`: Save best DFA
- `--verbose`: Verbose output

### Example
To run the genetic algorithm with 50 states, 100 generations, and tournament selection:
```bash
python main.py --n_states 50 --n_generations 100 --selection_method tournament --tournament_size 5
```

To use image chunks instead of full images:
```bash
python main.py --use_chunks --chunk_size 7 7 --chunk_stride 3 3
```

## Project Structure 📂
```
dfa-mnist/
├── main.py                      # Main script to run the genetic algorithm
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
├── results/                     # Directory for storing results
└── src/                         # Source code
    ├── automata/                # Automata implementations
    │   ├── dfa.py               # DFA implementation
    │   └── dfa_ga.py            # DFA-specific genetic operators
    ├── data/                    # Data handling
    │   └── mnist_loader.py      # MNIST data loader
    └── genetic_algorithm/       # Genetic algorithm framework
        ├── base.py              # Base classes for genetic algorithm
        ├── config.py            # Configuration management
        ├── crossover.py         # Crossover operators
        ├── factory.py           # Factory for creating GA components
        ├── initialization.py    # Population initialization
        ├── logger.py            # Logging utilities
        ├── metrics.py           # Metrics for tracking GA progress
        ├── mutation.py          # Mutation operators
        └── selection.py         # Selection operators
```

## How It Works 🧠

### Deterministic Finite Automaton (DFA)
A DFA is a mathematical model of computation that processes a sequence of symbols and transitions between states based on the input. In this project, we use DFAs to process MNIST images, where pixel values are treated as input symbols.

### Genetic Algorithm
The genetic algorithm evolves a population of DFAs to improve their ability to recognize MNIST digits. The key components include:
1. **Chromosome Representation**: Each chromosome encodes a DFA, including its transition matrix, state types, and final state mapping.
2. **Fitness Evaluation**: Chromosomes are evaluated based on their accuracy in classifying MNIST digits.
3. **Selection**: Parents are selected based on their fitness using various selection methods (tournament, roulette wheel, etc.).
4. **Crossover**: Genetic material is exchanged between parents to create offspring.
5. **Mutation**: Random changes are introduced to maintain genetic diversity.

### Chunk-Based Processing
To handle the high dimensionality of MNIST images, the DFA can process images in chunks. This approach:
- Reduces the complexity of the input sequence
- Allows the DFA to focus on local patterns
- Improves computational efficiency

## Results 📊
The results of the genetic algorithm are saved in the `results` directory, including:
- Training progress visualization
- Best DFA model
- Performance metrics (accuracy, fitness, etc.)

## License 📜
This project is licensed under the MIT License. Feel free to use this project for personal or commercial purposes.

