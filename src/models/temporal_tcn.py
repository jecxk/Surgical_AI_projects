"""
Multi-Stage Temporal Convolutional Network (MS-TCN) for Surgical Phase Recognition.

Based on: "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DilatedResidualLayer(nn.Module):
    """Single dilated residual convolutional layer."""
    
    def __init__(self, dilation: int, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.norm(out)
        out = self.dropout(out)
        return x + out


class SingleStageTCN(nn.Module):
    """Single stage of TCN with dilated convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 10,
        num_filters: int = 64,
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, num_filters, 1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(2 ** i, num_filters, num_filters)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_filters, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MultiStageTCN(nn.Module):
    """
    Multi-Stage TCN for temporal action segmentation.
    
    Args:
        input_dim: Input feature dimension from backbone
        num_stages: Number of refinement stages
        num_layers_per_stage: Number of dilated conv layers per stage
        num_filters: Number of convolutional filters
        num_classes: Number of output classes (for internal refinement)
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        num_stages: int = 3,
        num_layers_per_stage: int = 10,
        num_filters: int = 64,
        num_classes: int = 7,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.output_dim = num_filters
        
        # First stage: features -> predictions
        self.stage1 = SingleStageTCN(
            in_channels=input_dim,
            out_channels=num_classes,
            num_layers=num_layers_per_stage,
            num_filters=num_filters,
        )
        
        # Refinement stages: predictions -> refined predictions
        self.stages = nn.ModuleList([
            SingleStageTCN(
                in_channels=num_classes,
                out_channels=num_classes,
                num_layers=num_layers_per_stage,
                num_filters=num_filters,
            )
            for _ in range(num_stages - 1)
        ])
        
        # Feature output for multi-task head
        self.feature_proj = nn.Sequential(
            nn.Conv1d(num_classes, num_filters, 1),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor, return_all_stages: bool = False):
        """
        Args:
            x: Input features (B, T, input_dim)
            return_all_stages: Whether to return predictions from all stages
            
        Returns:
            features: (B, T, output_dim) for multi-task head
            stage_outputs: List of (B, T, num_classes) if return_all_stages
        """
        # TCN expects (B, C, T)
        x = x.permute(0, 2, 1)
        
        stage_outputs = []
        out = self.stage1(x)
        stage_outputs.append(out.permute(0, 2, 1))
        
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            stage_outputs.append(out.permute(0, 2, 1))
        
        # Get features for multi-task head
        features = self.feature_proj(out)  # (B, num_filters, T)
        features = features.permute(0, 2, 1)  # (B, T, num_filters)
        
        if return_all_stages:
            return features, stage_outputs
        return features, None
