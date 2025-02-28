import numpy as np
import cv2
from scipy import ndimage
from sklearn.preprocessing import StandardScaler

class MNISTPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
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
    
    def _normalize(self, images):
        """Normalize images to zero mean and unit variance"""
        # Reshape to 2D array for StandardScaler
        shape = images.shape
        flattened = images.reshape(shape[0], -1)
        
        # Fit and transform
        normalized = self.scaler.fit_transform(flattened)
        
        # Reshape back to original shape
        return normalized.reshape(shape)
    
    def _reduce_noise(self, images):
        """Apply Gaussian filter for noise reduction"""
        return np.array([cv2.GaussianBlur(img, (3, 3), 0.5) for img in images])
    
    def _binarize(self, images):
        """Binarize images using adaptive thresholding"""
        # Convert to uint8 and scale to 0-255 for cv2
        images_uint8 = (images * 255).astype(np.uint8)
        return np.array([
            cv2.adaptiveThreshold(
                img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            ).astype(np.float32)
            for img in images_uint8
        ])
    
    def _augment(self, images):
        """Apply data augmentation techniques"""
        height, width = images[0].shape
        n_samples = len(images)
        
        # Pre-allocate list with original images
        augmented = list(images)
        
        for i in range(n_samples):
            image = images[i]
            img_uint8 = (image * 255).astype(np.uint8)
            
            # Rotation
            if self.config.rotation_range > 0:
                angle = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)
                matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
                rotated = cv2.warpAffine(img_uint8, matrix, (width, height))
                # Ensure same shape
                rotated = cv2.resize(rotated, (width, height))
                augmented.append(rotated.astype(np.float32) / 255.0)
            
            # Shifts
            if self.config.width_shift > 0 or self.config.height_shift > 0:
                shift_x = np.random.uniform(-self.config.width_shift, self.config.width_shift) * width
                shift_y = np.random.uniform(-self.config.height_shift, self.config.height_shift) * height
                matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                shifted = cv2.warpAffine(img_uint8, matrix, (width, height))
                # Ensure same shape
                shifted = cv2.resize(shifted, (width, height))
                augmented.append(shifted.astype(np.float32) / 255.0)
            
            # Zoom
            if self.config.zoom_range > 0:
                zoom_factor = np.random.uniform(1 - self.config.zoom_range, 1 + self.config.zoom_range)
                
                # Calculate new dimensions
                new_height = int(height * zoom_factor)
                new_width = int(width * zoom_factor)
                
                # Perform zoom using resize
                zoomed = cv2.resize(img_uint8, (new_width, new_height))
                
                # Handle cropping or padding to maintain original size
                if zoom_factor > 1:
                    # Crop center
                    start_y = (new_height - height) // 2
                    start_x = (new_width - width) // 2
                    zoomed = zoomed[start_y:start_y + height, start_x:start_x + width]
                else:
                    # Pad with zeros
                    pad_y = (height - new_height) // 2
                    pad_x = (width - new_width) // 2
                    zoomed = cv2.copyMakeBorder(
                        zoomed, 
                        pad_y, height - new_height - pad_y,
                        pad_x, width - new_width - pad_x,
                        cv2.BORDER_CONSTANT, 
                        value=0
                    )
                
                # Final resize to ensure correct shape
                zoomed = cv2.resize(zoomed, (width, height))
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