import os
import gzip
import numpy as np
import urllib.request
from tqdm import tqdm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class MNISTLoader:
    def __init__(self, config):
        self.config = config
        self._load_data()

    def _load_data(self):
        """Load MNIST dataset using sklearn"""
        print("Loading MNIST dataset from OpenML...")
        mnist = fetch_openml('mnist_784', as_frame=False, parser='auto')
        
        # Reshape data to (n_samples, height, width)
        self.X = mnist.data.reshape(-1, 28, 28) / 255.0  # Normalize to [0, 1]
        self.y = mnist.target.astype(np.int32)
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=0.2,
            random_state=self.config.random_seed,
            stratify=self.y
        )
        
        print(f"Dataset loaded: {len(self.X_train)} training samples, {len(self.X_test)} test samples")

    def load_training_data(self):
        """Get training data"""
        return self.X_train, self.y_train

    def load_test_data(self):
        """Get test data"""
        return self.X_test, self.y_test 