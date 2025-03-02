import os
import gzip
import numpy as np
import urllib.request
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any, List

class MNISTLoader:
    """
    Class for loading and preprocessing the MNIST dataset.
    
    This class provides methods for loading the MNIST dataset, preprocessing it,
    and splitting it into training, validation, and test sets.
    """
    
    def __init__(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_state: int = 42,
        normalize: bool = True,
        binarize: bool = False,
        binarize_threshold: float = 0.5,
        flatten: bool = True,
        limit_samples: Optional[int] = None
    ):
        """
        Initialize the MNIST loader.
        
        Args:
            test_size: The proportion of the dataset to include in the test split.
            validation_size: The proportion of the training set to include in the validation split.
            random_state: Random state for reproducibility.
            normalize: Whether to normalize the data to [0, 1].
            binarize: Whether to binarize the data.
            binarize_threshold: Threshold for binarization.
            flatten: Whether to flatten the images.
            limit_samples: Limit the number of samples to load (for debugging).
        """
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_state = random_state
        self.normalize = normalize
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.flatten = flatten
        self.limit_samples = limit_samples
        
        # Initialize data attributes
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        # Initialize preprocessing attributes
        self.scaler = MinMaxScaler() if normalize else None
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the MNIST dataset and split it into training, validation, and test sets.
        
        Returns:
            A tuple containing (X_train, y_train, X_val, y_val, X_test, y_test).
        """
        print("Loading MNIST dataset...")
        
        # Load MNIST dataset
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        
        # Convert labels to integers
        y = y.astype(int)
        
        # Limit samples if specified
        if self.limit_samples is not None:
            X = X[:self.limit_samples]
            y = y[:self.limit_samples]
        
        # Determine if we should use stratification based on sample size
        # For very small sample sizes, stratification might not be possible
        use_stratify = True
        if self.limit_samples is not None and self.limit_samples < 100:
            # Check if we have at least 2 samples per class
            unique_labels, counts = np.unique(y, return_counts=True)
            if np.min(counts) < 2:
                use_stratify = False
                print("Warning: Sample size too small for stratification. Disabling stratified sampling.")
        
        # Split into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y if use_stratify else None
        )
        
        # Split train+val into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=self.validation_size, 
            random_state=self.random_state,
            stratify=y_train_val if use_stratify else None
        )
        
        # Preprocess data
        X_train, X_val, X_test = self._preprocess_data(X_train, X_val, X_test)
        
        # Store data
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"Data loaded and preprocessed:")
        print(f"  Training set: {X_train.shape}")
        print(f"  Validation set: {X_val.shape}")
        print(f"  Test set: {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _preprocess_data(
        self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the data.
        
        Args:
            X_train: Training data.
            X_val: Validation data.
            X_test: Test data.
            
        Returns:
            Preprocessed data.
        """
        # Normalize data
        if self.normalize:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
        
        # Binarize data
        if self.binarize:
            X_train = (X_train > self.binarize_threshold).astype(float)
            X_val = (X_val > self.binarize_threshold).astype(float)
            X_test = (X_test > self.binarize_threshold).astype(float)
        
        # Reshape data if not flattened
        if not self.flatten:
            X_train = X_train.reshape(-1, 28, 28)
            X_val = X_val.reshape(-1, 28, 28)
            X_test = X_test.reshape(-1, 28, 28)
        
        return X_train, X_val, X_test
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the loaded data.
        
        Returns:
            A tuple containing (X_train, y_train, X_val, y_val, X_test, y_test).
        """
        if self.X_train is None:
            return self.load_data()
        
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
    
    def get_sample_by_digit(self, digit: int, dataset: str = 'train') -> np.ndarray:
        """
        Get a sample image for a specific digit.
        
        Args:
            digit: The digit to get a sample for (0-9).
            dataset: The dataset to get the sample from ('train', 'val', 'test').
            
        Returns:
            A sample image for the specified digit.
        """
        # Check if data is loaded
        if self.X_train is None:
            self.load_data()
        
        # Get data for the specified dataset
        if dataset == 'train':
            X, y = self.X_train, self.y_train
        elif dataset == 'val':
            X, y = self.X_val, self.y_val
        elif dataset == 'test':
            X, y = self.X_test, self.y_test
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        
        # Find samples for the specified digit
        digit_indices = np.where(y == digit)[0]
        
        if len(digit_indices) == 0:
            raise ValueError(f"No samples found for digit {digit} in {dataset} dataset")
        
        # Get a random sample
        sample_idx = np.random.choice(digit_indices)
        sample = X[sample_idx]
        
        # Reshape if flattened
        if self.flatten:
            sample = sample.reshape(28, 28)
        
        return sample
    
    def visualize_samples(self, n_samples: int = 10, dataset: str = 'train') -> None:
        """
        Visualize random samples from the dataset.
        
        Args:
            n_samples: Number of samples to visualize.
            dataset: The dataset to visualize samples from ('train', 'val', 'test').
        """
        # Check if data is loaded
        if self.X_train is None:
            self.load_data()
        
        # Get data for the specified dataset
        if dataset == 'train':
            X, y = self.X_train, self.y_train
        elif dataset == 'val':
            X, y = self.X_val, self.y_val
        elif dataset == 'test':
            X, y = self.X_test, self.y_test
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        
        # Select random samples
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        # Create figure
        fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2, 2))
        
        for i, idx in enumerate(indices):
            # Get sample
            sample = X[idx]
            label = y[idx]
            
            # Reshape if flattened
            if self.flatten:
                sample = sample.reshape(28, 28)
            
            # Plot sample
            axes[i].imshow(sample, cmap='gray')
            axes[i].set_title(f"Digit: {label}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_digit_samples(self, dataset: str = 'train') -> None:
        """
        Visualize one sample for each digit.
        
        Args:
            dataset: The dataset to visualize samples from ('train', 'val', 'test').
        """
        # Create figure
        fig, axes = plt.subplots(1, 10, figsize=(20, 2))
        
        for digit in range(10):
            # Get sample for this digit
            sample = self.get_sample_by_digit(digit, dataset)
            
            # Plot sample
            axes[digit].imshow(sample, cmap='gray')
            axes[digit].set_title(f"Digit: {digit}")
            axes[digit].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_chunk(
        self, 
        image: np.ndarray, 
        chunk_size: Tuple[int, int], 
        chunk_stride: Tuple[int, int],
        flatten: bool = True
    ) -> List[np.ndarray]:
        """
        Split an image into chunks.
        
        Args:
            image: The image to split.
            chunk_size: The size of each chunk (height, width).
            chunk_stride: The stride for extracting chunks (height, width).
            flatten: Whether to flatten the chunks.
            
        Returns:
            A list of chunks.
        """
        # Reshape if flattened
        if self.flatten and image.ndim == 1:
            image = image.reshape(28, 28)
        
        # Get image dimensions
        height, width = image.shape
        
        # Get chunk dimensions
        chunk_height, chunk_width = chunk_size
        stride_height, stride_width = chunk_stride
        
        # Calculate number of chunks
        n_chunks_height = (height - chunk_height) // stride_height + 1
        n_chunks_width = (width - chunk_width) // stride_width + 1
        
        # Extract chunks
        chunks = []
        
        for i in range(n_chunks_height):
            for j in range(n_chunks_width):
                # Calculate chunk coordinates
                start_h = i * stride_height
                end_h = start_h + chunk_height
                start_w = j * stride_width
                end_w = start_w + chunk_width
                
                # Extract chunk
                chunk = image[start_h:end_h, start_w:end_w]
                
                # Flatten chunk if requested
                if flatten:
                    chunk = chunk.flatten()
                
                chunks.append(chunk)
        
        return chunks
    
    def get_chunks_for_dataset(
        self, 
        X: np.ndarray, 
        chunk_size: Tuple[int, int], 
        chunk_stride: Tuple[int, int],
        flatten: bool = True
    ) -> List[List[np.ndarray]]:
        """
        Split a dataset into chunks.
        
        Args:
            X: The dataset to split.
            chunk_size: The size of each chunk (height, width).
            chunk_stride: The stride for extracting chunks (height, width).
            flatten: Whether to flatten the chunks.
            
        Returns:
            A list of lists of chunks, one list per image.
        """
        # Extract chunks for each image
        chunks = []
        
        for image in tqdm(X, desc="Extracting chunks"):
            image_chunks = self.get_chunk(image, chunk_size, chunk_stride, flatten)
            chunks.append(image_chunks)
        
        return chunks 