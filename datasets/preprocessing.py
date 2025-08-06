import torch
import torch.nn.functional as F
import cv2
import numpy as np
from skimage import exposure, restoration

class AdaptivePreprocessor:
    def __init__(self, enable_hist_eq=True, enable_edge_preserving=True):
        self.enable_hist_eq = enable_hist_eq
        self.enable_edge_preserving = enable_edge_preserving
    
    def adaptive_histogram_equalization(self, image):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image
        
        # Apply CLAHE per channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = np.zeros_like(image_np)
        
        for i in range(image_np.shape[0]):  # channels
            enhanced[i] = clahe.apply((image_np[i] * 255).astype(np.uint8)) / 255.0
        
        return torch.from_numpy(enhanced).to(image.device)
    
    def edge_preserving_denoising(self, image, sigma_color=0.2, sigma_space=0.2):
        """Apply anisotropic diffusion for edge-preserving denoising"""
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image
        
        # Apply bilateral filter for edge preservation
        denoised = np.zeros_like(image_np)
        for i in range(image_np.shape[0]):
            denoised[i] = restoration.denoise_bilateral(
                image_np[i], 
                sigma_color=sigma_color, 
                sigma_spatial=sigma_space
            )
        
        return torch.from_numpy(denoised).to(image.device)
    
    def __call__(self, image):
        if self.enable_hist_eq:
            image = self.adaptive_histogram_equalization(image)
        if self.enable_edge_preserving:
            image = self.edge_preserving_denoising(image)
        return image
