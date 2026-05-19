"""
Multi-Task Prediction Heads for Phase Recognition and Tool Detection.
"""

import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head for simultaneous phase recognition
    and surgical tool detection.
    
    Args:
        input_dim: Input feature dimension from temporal model
        num_phases: Number of surgical phases
        num_tools: Number of tool classes
        dropout: Dropout rate
        use_tool_head: Whether to include tool detection head
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        num_phases: int = 7,
        num_tools: int = 7,
        dropout: float = 0.3,
        use_tool_head: bool = True,
    ):
        super().__init__()
        self.use_tool_head = use_tool_head
        
        # Phase recognition head
        self.phase_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_phases),
        )
        
        # Tool detection head (multi-label binary classification)
        if use_tool_head:
            self.tool_head = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_tools),
            )
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Features from temporal model (B, T, input_dim) or (B, input_dim)
            
        Returns:
            phase_logits: (B, T, num_phases) or (B, num_phases)
            tool_logits: (B, T, num_tools) or (B, num_tools), or None
        """
        phase_logits = self.phase_head(x)
        tool_logits = self.tool_head(x) if self.use_tool_head else None
        
        return phase_logits, tool_logits
