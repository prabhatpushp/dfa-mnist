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
        """Normalize images to zero mean and unit variance per image"""
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
                norm_img = (norm_img - norm_img.min()) / (norm_img.max() - norm_img.min())
            normalized.append(norm_img)
        return np.array(normalized)
    
    def _reduce_noise(self, images):
        """Apply Gaussian filter for noise reduction"""
        # Ensure input is in [0, 1] range
        images = np.clip(images, 0, 1)
        # Convert to uint8, apply filter, and convert back to float32
        images_uint8 = (images * 255).astype(np.uint8)
        denoised = np.array([cv2.GaussianBlur(img, (3, 3), 0.5) for img in images_uint8])
        return denoised.astype(np.float32) / 255.0
    
    def _binarize(self, images):
        """Binarize images using adaptive thresholding"""
        # Ensure input is in [0, 1] range
        images = np.clip(images, 0, 1)
        # Convert to uint8 and scale to 0-255 for cv2
        images_uint8 = (images * 255).astype(np.uint8)
        binarized = []
        
        for img in images_uint8:
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            binarized.append(binary)
        
        result = np.array(binarized).astype(np.float32) / 255.0
        # Invert the images so digits are white on black background
        return 1 - result
    
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