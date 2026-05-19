"""
LSTM-based Temporal Model for Surgical Phase Recognition.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class TemporalLSTM(nn.Module):
    """
    Bidirectional LSTM for temporal modeling of surgical phases.
    
    Takes frame-level features and models temporal dependencies
    to predict surgical phases for each timestep.
    
    Args:
        input_dim: Dimension of input features (from backbone)
        hidden_dim: LSTM hidden state dimension
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Output dimension
        self.output_dim = hidden_dim * self.num_directions
        
        # Output normalization
        self.output_norm = nn.LayerNorm(self.output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input features (B, T, input_dim)
            hidden: Optional initial hidden state
            
        Returns:
            output: Temporal features (B, T, output_dim)
            hidden: Final hidden state
        """
        # Project input
        x = self.input_proj(x)  # (B, T, hidden_dim)
        
        # LSTM forward
        output, hidden = self.lstm(x, hidden)  # (B, T, hidden_dim * num_directions)
        
        # Normalize output
        output = self.output_norm(output)
        
        return output, hidden
