import torch
import numpy as np

class AdaptiveSampler:
    def __init__(self, total_steps=100, quality_threshold=0.8):
        self.total_steps = total_steps
        self.quality_threshold = quality_threshold
        
    def compute_image_quality(self, image):
        """Compute local image quality metrics - FIXED VERSION"""
        # Compute gradients
        grad_x = torch.diff(image, dim=3)  # Shape: [B, C, H, W-1]
        grad_y = torch.diff(image, dim=2)  # Shape: [B, C, H-1, W]
        
        # Fix dimension mismatch by properly aligning gradients
        # Take the overlapping region for both gradients
        grad_x_aligned = grad_x[:, :, :-1, :]  # [B, C, H-1, W-1]
        grad_y_aligned = grad_y[:, :, :, :-1]  # [B, C, H-1, W-1]
        
        # Now both tensors have the same shape: [B, C, H-1, W-1]
        gradient_magnitude = torch.sqrt(grad_x_aligned**2 + grad_y_aligned**2)
        quality_score = torch.mean(gradient_magnitude)
        return quality_score
    
    def dynamic_timestep_scheduling(self, current_step, image_quality):
        """Adjust timestep based on current restoration quality"""
        if image_quality < self.quality_threshold:
            # Use smaller timesteps for better quality
            timestep_scale = 0.8
        else:
            # Use larger timesteps for efficiency
            timestep_scale = 1.2
        
        return int(timestep_scale * (self.total_steps - current_step))
    
    def proximal_sampling(self, predicted_samples, measurement, num_candidates=5):
        """Select best sample from candidates based on data consistency"""
        best_sample = None
        best_consistency = float('inf')
        
        for sample in predicted_samples[:num_candidates]:
            # Compute data consistency error
            consistency_error = torch.norm(sample - measurement)
            
            if consistency_error < best_consistency:
                best_consistency = consistency_error
                best_sample = sample
        
        return best_sample
