import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import sobel
import numpy as np

class MultiLevelEdgeGuidance(nn.Module):
    def __init__(self, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.edge_detectors = nn.ModuleList([
            nn.Conv2d(3, 1, 3, padding=1) for _ in range(num_levels)
        ])
        
    def extract_edges(self, image, level=0):
        """Extract multi-scale edge information"""
        if level > 0:
            # Downsample for multi-scale processing
            h, w = image.shape[2], image.shape[3]
            scale = 2 ** level
            image = F.interpolate(image, size=(h//scale, w//scale), mode='bilinear')
        
        # Apply learned edge detection
        edges = self.edge_detectors[level](image)
        
        if level > 0:
            # Upsample back to original resolution
            edges = F.interpolate(edges, size=(h, w), mode='bilinear')
        
        return edges
    
    def forward(self, image):
        multi_level_edges = []
        for level in range(self.num_levels):
            edges = self.extract_edges(image, level)
            multi_level_edges.append(edges)
        
        # Combine multi-scale edges
        combined_edges = torch.cat(multi_level_edges, dim=1)
        return combined_edges

class FrequencyGuidance(nn.Module):
    def __init__(self):
        super().__init__()
        
    def frequency_decomposition(self, image):
        """Decompose image into frequency components"""
        fft_image = torch.fft.fft2(image)
        fft_shift = torch.fft.fftshift(fft_image)
        
        # Separate amplitude and phase
        amplitude = torch.abs(fft_shift)
        phase = torch.angle(fft_shift)
        
        return amplitude, phase
    
    def frequency_enhancement(self, amplitude, phase):
        """Enhance frequency components for better restoration"""
        # Apply frequency domain filtering
        enhanced_amplitude = amplitude * 1.2  # Boost high frequencies
        
        # Reconstruct image
        enhanced_fft = enhanced_amplitude * torch.exp(1j * phase)
        enhanced_fft = torch.fft.ifftshift(enhanced_fft)
        enhanced_image = torch.real(torch.fft.ifft2(enhanced_fft))
        
        return enhanced_image
    
    def forward(self, image):
        amplitude, phase = self.frequency_decomposition(image)
        enhanced_image = self.frequency_enhancement(amplitude, phase)
        return enhanced_image
