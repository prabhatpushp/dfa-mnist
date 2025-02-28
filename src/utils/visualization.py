import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class DataVisualizer:
    def __init__(self, config):
        self.config = config
        
    def plot_samples(self, images, labels, title, n_samples=5):
        """Plot sample images with their labels"""
        fig, axes = plt.subplots(1, n_samples, figsize=self.config.plot_size)
        
        for i, (image, label) in enumerate(zip(images[:n_samples], labels[:n_samples])):
            axes[i].imshow(image, cmap=self.config.cmap)
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
        
    def plot_preprocessing_steps(self, original, normalized, denoised, binarized):
        """Plot the results of each preprocessing step"""
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        
        steps = [
            ('Original', original),
            ('Normalized', normalized),
            ('Denoised', denoised),
            ('Binarized', binarized)
        ]
        
        for ax, (title, image) in zip(axes, steps):
            ax.imshow(image, cmap=self.config.cmap)
            ax.set_title(title)
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
        
    def plot_augmented_samples(self, original, augmented):
        """Plot original image alongside its augmented versions"""
        n_augmented = len(augmented)
        fig, axes = plt.subplots(1, n_augmented + 1, figsize=(3 * (n_augmented + 1), 3))
        
        axes[0].imshow(original, cmap=self.config.cmap)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        for i, aug_img in enumerate(augmented, 1):
            axes[i].imshow(aug_img, cmap=self.config.cmap)
            axes[i].set_title(f'Augmented {i}')
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    def plot_feature_distribution(self, features, labels):
        """Plot distribution of extracted features"""
        plt.figure(figsize=(10, 6))
        for digit in range(10):
            digit_features = features[labels == digit]
            plt.hist(digit_features.ravel(), bins=50, alpha=0.3, label=f'Digit {digit}')
            
        plt.title('Distribution of HOG Features by Digit')
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix of classification results"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
    def plot_learning_curve(self, train_scores, val_scores, epochs):
        """Plot learning curves during training"""
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_scores, label='Training Score')
        plt.plot(epochs, val_scores, label='Validation Score')
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.show() 