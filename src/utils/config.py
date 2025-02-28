class Config:
    def __init__(self):
        # Data parameters
        self.image_size = 28
        self.num_classes = 10
        self.random_seed = 42
        
        # Preprocessing parameters
        self.normalize = True
        self.binarize = True
        self.binarization_threshold = 0.5
        self.noise_reduction = True
        self.augmentation = True
        
        # Data augmentation parameters
        self.rotation_range = 15
        self.width_shift = 0.1
        self.height_shift = 0.1
        self.zoom_range = 0.1
        
        # Training parameters
        self.batch_size = 32
        self.validation_split = 0.2
        
        # Feature extraction parameters
        self.hog_orientations = 9
        self.hog_pixels_per_cell = (7, 7)
        self.hog_cells_per_block = (2, 2)
        
        # Visualization parameters
        self.plot_size = (10, 10)
        self.cmap = 'gray' 