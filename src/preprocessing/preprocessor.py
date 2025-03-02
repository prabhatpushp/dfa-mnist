import numpy as np
import cv2
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List, Dict, Any, Union

class MNISTPreprocessor:
    """
    Preprocessor for MNIST images.
    
    This class provides methods for preprocessing MNIST images, including
    normalization, noise reduction, binarization, and feature extraction.
    It also supports reducing the number of symbols in the alphabet.
    """
    
    def __init__(self, n_symbols: int = 256, normalize: bool = True, 
                 noise_reduction: bool = False, binarize: bool = False,
                 binarize_threshold: float = 0.5):
        """
        Initialize the MNIST preprocessor.
        
        Args:
            n_symbols: The number of symbols in the alphabet (default: 256).
            normalize: Whether to normalize the images (default: True).
            noise_reduction: Whether to apply noise reduction (default: False).
            binarize: Whether to binarize the images (default: False).
            binarize_threshold: The threshold for binarization (default: 0.5).
        """
        self.n_symbols = n_symbols
        self.normalize = normalize
        self.noise_reduction = noise_reduction
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.scaler = StandardScaler()
        self.symbol_bins = None
        
    def fit(self, images: np.ndarray) -> 'MNISTPreprocessor':
        """
        Fit the preprocessor to the data.
        
        Args:
            images: The images to fit the preprocessor to.
            
        Returns:
            The fitted preprocessor.
        """
        # Apply basic preprocessing
        processed = self._basic_preprocessing(images)
        
        # Determine symbol bins based on the distribution of pixel values
        if self.n_symbols < 256:
            # Flatten all images to get pixel value distribution
            flat_pixels = processed.reshape(-1)
            
            # Create histogram bins for quantization
            # Use percentiles to ensure even distribution of symbols
            percentiles = np.linspace(0, 100, self.n_symbols + 1)
            self.symbol_bins = np.percentile(flat_pixels, percentiles)
            
            # Ensure the last bin includes the maximum value
            self.symbol_bins[-1] = 1.0 + 1e-6
        
        return self
    
    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to the images.
        
        Args:
            images: The images to preprocess.
            
        Returns:
            The preprocessed images.
        """
        # Apply basic preprocessing
        processed = self._basic_preprocessing(images)
        
        # Reduce the number of symbols if needed
        if self.n_symbols < 256 and self.symbol_bins is not None:
            processed = self._reduce_symbols(processed)
        
        return processed
    
    def fit_transform(self, images: np.ndarray) -> np.ndarray:
        """
        Fit the preprocessor to the data and apply preprocessing.
        
        Args:
            images: The images to fit and preprocess.
            
        Returns:
            The preprocessed images.
        """
        self.fit(images)
        return self.transform(images)
    
    def _basic_preprocessing(self, images: np.ndarray) -> np.ndarray:
        """
        Apply basic preprocessing steps to the images.
        
        Args:
            images: The images to preprocess.
            
        Returns:
            The preprocessed images.
        """
        processed = images.copy()
        
        if self.normalize:
            processed = self._normalize(processed)
            
        if self.noise_reduction:
            processed = self._reduce_noise(processed)
            
        if self.binarize:
            processed = self._binarize(processed)
        
        return processed
    
    def _reduce_symbols(self, images: np.ndarray) -> np.ndarray:
        """
        Reduce the number of symbols in the images.
        
        Args:
            images: The images to reduce symbols in.
            
        Returns:
            The images with reduced symbols.
        """
        # Create output array with same shape
        reduced = np.zeros_like(images)
        
        # Apply quantization using the symbol bins
        for i in range(len(self.symbol_bins) - 1):
            mask = (images >= self.symbol_bins[i]) & (images < self.symbol_bins[i + 1])
            reduced[mask] = i / (self.n_symbols - 1)  # Normalize to [0, 1]
        
        return reduced
    
    def _normalize(self, images: np.ndarray) -> np.ndarray:
        """
        Normalize images to zero mean and unit variance per image.
        
        Args:
            images: The images to normalize.
            
        Returns:
            The normalized images.
        """
        normalized = []
        for image in images:
            # Normalize each image individually
            img_mean = np.mean(image)
            img_std = np.std(image)
            if img_std == 0:
                # Handle constant-value images
                norm_img = np.zeros_like(image)
            else:
                norm_img = (image - img_mean) / img_std
                # Scale to [0, 1] range
                min_val = norm_img.min()
                max_val = norm_img.max()
                if max_val > min_val:
                    norm_img = (norm_img - min_val) / (max_val - min_val)
                else:
                    norm_img = np.zeros_like(norm_img)
            normalized.append(norm_img)
        return np.array(normalized)
    
    def _reduce_noise(self, images: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian filter for noise reduction.
        
        Args:
            images: The images to reduce noise in.
            
        Returns:
            The noise-reduced images.
        """
        filtered = []
        for image in images:
            # Apply Gaussian filter
            filtered_img = ndimage.gaussian_filter(image, sigma=0.5)
            filtered.append(filtered_img)
        return np.array(filtered)
    
    def _binarize(self, images: np.ndarray) -> np.ndarray:
        """
        Binarize images using a threshold.
        
        Args:
            images: The images to binarize.
            
        Returns:
            The binarized images.
        """
        binarized = []
        for image in images:
            # Apply threshold
            bin_img = np.where(image > self.binarize_threshold, 1.0, 0.0)
            binarized.append(bin_img)
        return np.array(binarized)
    
    def visualize_symbol_reduction(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Visualize the effect of symbol reduction on an image.
        
        Args:
            image: The original image.
            
        Returns:
            A tuple containing the original and reduced images.
        """
        # Apply basic preprocessing
        processed = self._basic_preprocessing(image[np.newaxis, ...])[0]
        
        # Reduce symbols
        if self.n_symbols < 256 and self.symbol_bins is not None:
            reduced = self._reduce_symbols(processed[np.newaxis, ...])[0]
        else:
            reduced = processed
        
        return processed, reduced
    
    def preprocess(self, images):
        """Apply all preprocessing steps to the images"""
        processed = images.copy()
        
        if self.config.normalize:
            processed = self._normalize(processed)
            
        if self.config.noise_reduction:
            processed = self._reduce_noise(processed)
            
        if self.config.binarize:
            processed = self._binarize(processed)
            
        # Skip augmentation during full preprocessing
        # as it would create multiple versions of each image
        
        # Extract HOG features
        processed = self._extract_hog_features(processed)
        
        return processed
    
    def _augment(self, images):
        """Apply data augmentation techniques"""
        height, width = images[0].shape
        n_samples = len(images)
        
        # Pre-allocate list with original images
        augmented = list(images)
        
        for i in range(n_samples):
            image = images[i]
            # Scale to 0-255 and convert to uint8
            img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
            
            # Rotation
            if self.config.rotation_range > 0:
                angle = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)
                matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
                rotated = cv2.warpAffine(img_uint8, matrix, (width, height), 
                                       borderMode=cv2.BORDER_REPLICATE)
                augmented.append(rotated.astype(np.float32) / 255.0)
            
            # Shifts
            if self.config.width_shift > 0 or self.config.height_shift > 0:
                shift_x = np.random.uniform(-self.config.width_shift, self.config.width_shift) * width
                shift_y = np.random.uniform(-self.config.height_shift, self.config.height_shift) * height
                matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                shifted = cv2.warpAffine(img_uint8, matrix, (width, height),
                                       borderMode=cv2.BORDER_REPLICATE)
                augmented.append(shifted.astype(np.float32) / 255.0)
            
            # Zoom
            if self.config.zoom_range > 0:
                zoom_factor = np.random.uniform(1 - self.config.zoom_range, 1 + self.config.zoom_range)
                
                if zoom_factor > 1:  # Zoom in
                    # Crop center
                    crop_size = int(min(width, height) / zoom_factor)
                    start_y = (height - crop_size) // 2
                    start_x = (width - crop_size) // 2
                    cropped = img_uint8[start_y:start_y + crop_size, start_x:start_x + crop_size]
                    zoomed = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
                else:  # Zoom out
                    # Resize to smaller size
                    new_size = int(min(width, height) * zoom_factor)
                    small = cv2.resize(img_uint8, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
                    # Pad with border replication
                    pad_size = (width - new_size) // 2
                    zoomed = cv2.copyMakeBorder(small, pad_size, pad_size, pad_size, pad_size,
                                              cv2.BORDER_REPLICATE)
                    # Ensure correct size
                    zoomed = cv2.resize(zoomed, (width, height), interpolation=cv2.INTER_LINEAR)
                
                augmented.append(zoomed.astype(np.float32) / 255.0)
        
        # Verify all images have the same shape before stacking
        shapes = [img.shape for img in augmented]
        if len(set(shapes)) > 1:
            raise ValueError(f"Inconsistent shapes in augmented images: {shapes}")
        
        # Stack all augmented images
        return np.stack(augmented)
    
    def _extract_hog_features(self, images):
        """Extract Histogram of Oriented Gradients (HOG) features using OpenCV"""
        features = []
        
        # Calculate HOG parameters
        win_size = (self.config.image_size, self.config.image_size)
        cell_size = self.config.hog_pixels_per_cell
        block_size = tuple(b * c for b, c in zip(self.config.hog_cells_per_block, cell_size))
        block_stride = (cell_size[0], cell_size[1])  # Use cell size as stride
        
        # Verify parameters
        if (win_size[0] - block_size[0]) % block_stride[0] != 0 or \
           (win_size[1] - block_size[1]) % block_stride[1] != 0:
            raise ValueError(
                f"Invalid HOG parameters: window size {win_size}, "
                f"block size {block_size}, stride {block_stride}"
            )
        
        try:
            # Initialize HOG descriptor
            hog = cv2.HOGDescriptor(
                win_size,
                block_size,
                block_stride,
                cell_size,
                self.config.hog_orientations
            )
            
            # Process each image
            for image in images:
                # Convert to uint8 for cv2
                img_uint8 = (image * 255).astype(np.uint8)
                # Compute HOG features
                hog_features = hog.compute(img_uint8)
                if hog_features is None:
                    raise ValueError("HOG feature computation failed")
                features.append(hog_features.flatten())
                
        except cv2.error as e:
            raise ValueError(f"HOG computation error: {str(e)}")
            
        return np.array(features) 