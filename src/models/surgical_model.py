"""
Full Surgical Phase Recognition Model.

Combines backbone + temporal model + multi-task prediction heads
into a unified end-to-end trainable model.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .backbone import BackboneFeatureExtractor, get_backbone
from .temporal_lstm import TemporalLSTM
from .temporal_tcn import MultiStageTCN
from .temporal_transformer import TemporalTransformer
from .multi_task_head import MultiTaskHead


class SurgicalPhaseModel(nn.Module):
    """
    End-to-end Surgical Phase Recognition Model.
    
    Architecture: Backbone -> Temporal Model -> Multi-Task Heads
    
    Args:
        config: Model configuration dictionary
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        backbone_name = config.get('backbone', 'resnet50')
        temporal_type = config.get('temporal_model', 'lstm')
        feature_dim = config.get('feature_dim', 2048)
        dropout = config.get('dropout', 0.3)
        num_phases = config.get('num_phases', 7)
        num_tools = config.get('num_tools', 7)
        use_tools = config.get('tool_detection', True)
        
        # 1. Backbone feature extractor
        if temporal_type != 'timesformer':
            self.backbone = get_backbone(backbone_name, pretrained=config.get('backbone_pretrained', True))
            actual_feature_dim = self.backbone.feature_dim
        else:
            self.backbone = None
            actual_feature_dim = feature_dim
        
        # 2. Temporal model
        if temporal_type == 'lstm':
            lstm_cfg = config.get('lstm', {})
            self.temporal = TemporalLSTM(
                input_dim=actual_feature_dim,
                hidden_dim=lstm_cfg.get('hidden_dim', 512),
                num_layers=lstm_cfg.get('num_layers', 2),
                dropout=dropout,
                bidirectional=lstm_cfg.get('bidirectional', True),
            )
            temporal_out_dim = self.temporal.output_dim
            
        elif temporal_type == 'tcn':
            tcn_cfg = config.get('tcn', {})
            self.temporal = MultiStageTCN(
                input_dim=actual_feature_dim,
                num_stages=tcn_cfg.get('num_stages', 3),
                num_layers_per_stage=tcn_cfg.get('num_layers_per_stage', 10),
                num_filters=tcn_cfg.get('num_filters', 64),
                num_classes=num_phases,
            )
            temporal_out_dim = self.temporal.output_dim
            
        elif temporal_type == 'transformer':
            tf_cfg = config.get('transformer', {})
            self.temporal = TemporalTransformer(
                input_dim=actual_feature_dim,
                d_model=tf_cfg.get('d_model', 512),
                num_heads=tf_cfg.get('num_heads', 8),
                num_layers=tf_cfg.get('num_layers', 4),
                dim_feedforward=tf_cfg.get('dim_feedforward', 1024),
                dropout=dropout,
                max_seq_length=tf_cfg.get('max_seq_length', 2048),
            )
            temporal_out_dim = self.temporal.output_dim
            
        else:
            raise ValueError(f"Unknown temporal model: {temporal_type}")
        
        # 3. Multi-task prediction heads
        self.heads = MultiTaskHead(
            input_dim=temporal_out_dim,
            num_phases=num_phases,
            num_tools=num_tools,
            dropout=dropout,
            use_tool_head=use_tools,
        )
        
        # TCN has its own phase predictions
        self.is_tcn = temporal_type == 'tcn'
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full model.
        
        Args:
            images: Input tensor (B, T, C, H, W)
            
        Returns:
            Dictionary with:
                - 'phase_logits': (B, T, num_phases)
                - 'tool_logits': (B, T, num_tools) or None
                - 'features': (B, T, feature_dim)
                - 'tcn_stages': list of stage outputs (TCN only)
        """
        B, T = images.shape[:2]
        
        # Extract spatial features
        if self.backbone is not None:
            features = self.backbone(images)  # (B, T, feature_dim)
        else:
            features = images  # For TimeSformer, features come directly
        
        # Temporal modeling
        if self.is_tcn:
            temporal_features, tcn_stages = self.temporal(features, return_all_stages=True)
        elif isinstance(self.temporal, TemporalLSTM):
            temporal_features, _ = self.temporal(features)
            tcn_stages = None
        else:
            temporal_features = self.temporal(features)
            tcn_stages = None
        
        # Multi-task prediction
        phase_logits, tool_logits = self.heads(temporal_features)
        
        output = {
            'phase_logits': phase_logits,
            'tool_logits': tool_logits,
            'features': features,
        }
        
        if tcn_stages is not None:
            output['tcn_stages'] = tcn_stages
        
        return output
    
    def freeze_backbone(self):
        """Freeze backbone for staged training."""
        if self.backbone is not None:
            self.backbone.freeze_backbone()
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for end-to-end fine-tuning."""
        if self.backbone is not None:
            self.backbone.unfreeze_backbone()
    
    def get_backbone_params(self):
        """Get backbone parameters for differential learning rate."""
        if self.backbone is not None:
            return self.backbone.parameters()
        return []
    
    def get_non_backbone_params(self):
        """Get non-backbone parameters."""
        params = list(self.temporal.parameters()) + list(self.heads.parameters())
        return params


def build_model(config: dict) -> SurgicalPhaseModel:
    """Factory function to build the surgical phase recognition model."""
    return SurgicalPhaseModel(config)
