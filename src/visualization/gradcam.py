"""
Grad-CAM Visualization for Surgical Phase Recognition.

Generates heatmaps showing what spatial regions the model
focuses on for each surgical phase prediction.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class SurgicalGradCAM:
    """
    Grad-CAM implementation for surgical phase recognition models.
    
    Args:
        model: Trained SurgicalPhaseModel
        target_layer: Target convolutional layer for Grad-CAM
        device: Computation device
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: torch.device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for an input image.
        
        Args:
            image: Input image tensor (C, H, W) or (1, C, H, W)
            target_class: Target class for Grad-CAM (None = predicted class)
            
        Returns:
            heatmap: Grad-CAM heatmap (H, W) normalized to [0, 1]
            predicted_class: Model's predicted class
            confidence: Prediction confidence
        """
        self.model.eval()
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # For sequence model, wrap in temporal dim
        if image.dim() == 4:
            image = image.unsqueeze(1)  # (1, 1, C, H, W)
        
        image = image.to(self.device)
        image.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(image)
        phase_logits = outputs['phase_logits']
        
        if phase_logits.dim() == 3:
            phase_logits = phase_logits[:, -1, :]  # Last timestep
        
        probs = F.softmax(phase_logits, dim=-1)
        predicted_class = probs.argmax(dim=-1).item()
        confidence = probs.max().item()
        
        if target_class is None:
            target_class = predicted_class
        
        # Backward pass
        self.model.zero_grad()
        target_score = phase_logits[0, target_class]
        target_score.backward()
        
        # Compute Grad-CAM
        if self.gradients is not None and self.activations is not None:
            gradients = self.gradients
            activations = self.activations
            
            if gradients.dim() == 5:
                gradients = gradients[:, 0]
                activations = activations[:, 0]
            
            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            
            # Weighted combination
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            
            # Normalize
            cam = cam.squeeze().cpu().numpy()
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Resize to input size
            cam = cv2.resize(cam, (image.shape[-1], image.shape[-2]))
        else:
            cam = np.zeros((image.shape[-2], image.shape[-1]))
        
        return cam, predicted_class, confidence
    
    def visualize(
        self,
        image: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Generate and overlay Grad-CAM visualization.
        
        Args:
            image: Preprocessed image tensor
            original_image: Original image (H, W, 3) in RGB, range [0, 255]
            target_class: Target class
            class_names: List of class names
            save_path: Path to save the visualization
            alpha: Overlay transparency
            
        Returns:
            Overlay image (H, W, 3)
        """
        cam, pred_class, confidence = self.generate(image, target_class)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Resize original image to match
        h, w = cam.shape
        original_resized = cv2.resize(original_image, (w, h))
        
        # Overlay
        overlay = (alpha * heatmap + (1 - alpha) * original_resized).astype(np.uint8)
        
        if save_path:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(original_resized)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            axes[1].imshow(cam, cmap='jet')
            axes[1].set_title("Grad-CAM Heatmap")
            axes[1].axis('off')
            
            pred_name = class_names[pred_class] if class_names else str(pred_class)
            axes[2].imshow(overlay)
            axes[2].set_title(f"Overlay\nPred: {pred_name} ({confidence:.2%})")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        return overlay
