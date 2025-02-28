import os
import numpy as np
from src.data.mnist_loader import MNISTLoader
from src.preprocessing.preprocessor import MNISTPreprocessor
from src.utils.visualization import DataVisualizer
from src.utils.config import Config

def main():
    # Initialize configuration
    config = Config()
    
    # Initialize MNIST data loader
    mnist_loader = MNISTLoader(config)
    
    # Load data
    X_train, y_train = mnist_loader.load_training_data()
    X_test, y_test = mnist_loader.load_test_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize preprocessor
    preprocessor = MNISTPreprocessor(config)
    
    # Initialize visualizer
    visualizer = DataVisualizer(config)
    
    # Show original samples
    print("\nDisplaying original samples...")
    visualizer.plot_samples(X_train[:5], y_train[:5], "Original MNIST Samples")
    
    # Show preprocessing steps for a single image
    print("\nDisplaying preprocessing steps...")
    sample_img = X_train[0]  # Shape: (28, 28)
    
    # Apply preprocessing steps one by one (keeping original shape)
    normalized = preprocessor._normalize(np.array([sample_img]))[0]  # Keep as (28, 28)
    denoised = preprocessor._reduce_noise(np.array([normalized]))[0]  # Keep as (28, 28)
    binarized = preprocessor._binarize(np.array([denoised]))[0]  # Keep as (28, 28)
    
    visualizer.plot_preprocessing_steps(
        sample_img,
        normalized,
        denoised,
        binarized
    )
    
    # Show augmentation examples
    print("\nDisplaying augmentation examples...")
    augmented = preprocessor._augment(np.array([sample_img]))
    visualizer.plot_augmented_samples(sample_img, augmented[1:])
    
    # Process subset of dataset first to verify
    print("\nProcessing subset of dataset...")
    subset_size = 1000
    X_train_subset = X_train[:subset_size]
    X_train_processed = preprocessor.preprocess(X_train_subset)
    
    print(f"Processed subset shape: {X_train_processed.shape}")
    
    # Show feature distribution for subset
    print("\nDisplaying feature distribution...")
    visualizer.plot_feature_distribution(X_train_processed, y_train[:subset_size])

if __name__ == "__main__":
    main()
