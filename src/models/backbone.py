"""
Backbone Feature Extractors for Surgical Phase Recognition.

Supports ResNet50, EfficientNet-B3, and Swin Transformer.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple


class BackboneFeatureExtractor(nn.Module):
    """
    CNN backbone that extracts spatial features from individual frames.
    
    Args:
        backbone_name: Name of the backbone architecture
        pretrained: Whether to use ImageNet pretrained weights
        freeze_layers: Number of initial layers to freeze (-1 = none)
    """
    
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_layers: int = -1,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base = models.resnet50(weights=weights)
            self.feature_dim = 2048
            # Remove the final FC layer
            self.features = nn.Sequential(*list(base.children())[:-1])
            self.target_layer_name = "features.7"  # layer4 for Grad-CAM
            
        elif backbone_name == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            base = models.resnet101(weights=weights)
            self.feature_dim = 2048
            self.features = nn.Sequential(*list(base.children())[:-1])
            self.target_layer_name = "features.7"
            
        elif backbone_name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.efficientnet_b3(weights=weights)
            self.feature_dim = 1536
            self.features = nn.Sequential(
                base.features,
                base.avgpool,
            )
            self.target_layer_name = "features.0.8"
            
        elif backbone_name == "swin_tiny":
            weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.swin_t(weights=weights)
            self.feature_dim = 768
            # Remove classification head
            self.features = nn.Sequential(
                base.features,
                base.norm,
                base.permute,
                base.avgpool,
                base.flatten,
            )
            self.target_layer_name = "features.0.7"
            
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Freeze early layers if specified
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
    
    def _freeze_layers(self, num_layers: int):
        """Freeze the first num_layers of the backbone."""
        children = list(self.features.children())
        for i, child in enumerate(children[:num_layers]):
            for param in child.parameters():
                param.requires_grad = False
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.features.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.
        
        Args:
            x: Input tensor of shape (B, C, H, W) or (B, T, C, H, W)
            
        Returns:
            Features of shape (B, feature_dim) or (B, T, feature_dim)
        """
        has_temporal = x.dim() == 5
        
        if has_temporal:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        if has_temporal:
            features = features.view(B, T, -1)
        
        return features
    
    def get_target_layer(self):
        """Get the target layer for Grad-CAM visualization."""
        parts = self.target_layer_name.split('.')
        layer = self
        for part in parts:
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        return layer


def get_backbone(name: str = "resnet50", pretrained: bool = True) -> BackboneFeatureExtractor:
    """Factory function to create a backbone feature extractor."""
    return BackboneFeatureExtractor(backbone_name=name, pretrained=pretrained)
