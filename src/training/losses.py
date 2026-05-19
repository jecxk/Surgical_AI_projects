"""
Loss Functions for Surgical Phase Recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class SurgicalLoss(nn.Module):
    """
    Combined loss for multi-task surgical phase recognition.
    
    Combines:
    - Phase classification loss (CrossEntropy with label smoothing)
    - Tool detection loss (Binary CrossEntropy)
    - Temporal consistency loss (optional)
    - TCN multi-stage loss (for MS-TCN)
    
    Args:
        num_phases: Number of surgical phases
        class_weights: Optional class weights for phase loss
        label_smoothing: Label smoothing factor
        tool_loss_weight: Weight for tool detection loss
        temporal_consistency_weight: Weight for temporal consistency
    """
    
    def __init__(
        self,
        num_phases: int = 7,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        tool_loss_weight: float = 0.5,
        temporal_consistency_weight: float = 0.1,
    ):
        super().__init__()
        self.tool_loss_weight = tool_loss_weight
        self.temporal_consistency_weight = temporal_consistency_weight
        
        # Phase classification loss
        self.phase_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        
        # Tool detection loss (multi-label)
        self.tool_loss = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            predictions: Model outputs with 'phase_logits', 'tool_logits'
            targets: Ground truth with 'phases', 'tools'
            
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Phase loss
        phase_logits = predictions['phase_logits']
        phase_targets = targets['phases']
        
        if phase_logits.dim() == 3:
            # (B, T, C) -> (B*T, C) for loss computation
            B, T, C = phase_logits.shape
            phase_logits_flat = phase_logits.reshape(-1, C)
            phase_targets_flat = phase_targets.reshape(-1)
            losses['phase_loss'] = self.phase_loss(phase_logits_flat, phase_targets_flat)
        else:
            losses['phase_loss'] = self.phase_loss(phase_logits, phase_targets)
        
        # Tool detection loss
        tool_logits = predictions.get('tool_logits')
        tool_targets = targets.get('tools')
        
        if tool_logits is not None and tool_targets is not None:
            if tool_logits.dim() == 3:
                B, T, N = tool_logits.shape
                tool_logits_flat = tool_logits.reshape(-1, N)
                tool_targets_flat = tool_targets.reshape(-1, N)
                losses['tool_loss'] = self.tool_loss(tool_logits_flat, tool_targets_flat)
            else:
                losses['tool_loss'] = self.tool_loss(tool_logits, tool_targets)
        else:
            losses['tool_loss'] = torch.tensor(0.0, device=phase_logits.device)
        
        # Temporal consistency loss (penalize rapid phase changes)
        if phase_logits.dim() == 3 and self.temporal_consistency_weight > 0:
            phase_probs = F.softmax(phase_logits, dim=-1)
            temporal_diff = torch.diff(phase_probs, dim=1)
            losses['temporal_loss'] = torch.mean(temporal_diff ** 2)
        else:
            losses['temporal_loss'] = torch.tensor(0.0, device=phase_logits.device)
        
        # TCN multi-stage loss
        tcn_stages = predictions.get('tcn_stages')
        if tcn_stages is not None:
            stage_loss = 0
            for stage_out in tcn_stages:
                if stage_out.dim() == 3:
                    B, T, C = stage_out.shape
                    stage_loss += self.phase_loss(
                        stage_out.reshape(-1, C), phase_targets.reshape(-1)
                    )
                else:
                    stage_loss += self.phase_loss(stage_out, phase_targets)
            losses['tcn_stage_loss'] = stage_loss / len(tcn_stages)
        
        # Total loss
        total = losses['phase_loss']
        total = total + self.tool_loss_weight * losses['tool_loss']
        total = total + self.temporal_consistency_weight * losses['temporal_loss']
        if 'tcn_stage_loss' in losses:
            total = total + losses['tcn_stage_loss']
        
        losses['total_loss'] = total
        return losses
