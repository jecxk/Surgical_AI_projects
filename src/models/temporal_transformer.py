"""
Temporal Transformer for Surgical Phase Recognition.
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for temporal modeling of surgical phases.
    
    Args:
        input_dim: Input feature dimension
        d_model: Transformer model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward network dimension
        dropout: Dropout rate
        max_seq_length: Maximum sequence length
    """
    
    def __init__(
        self,
        input_dim: int = 2048,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.3,
        max_seq_length: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = d_model
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output norm
        self.output_norm = nn.LayerNorm(d_model)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (B, T, input_dim)
            src_mask: Optional attention mask
            causal: Whether to use causal (autoregressive) mask
            
        Returns:
            output: Temporal features (B, T, d_model)
        """
        B, T, _ = x.shape
        
        # Project and add positional encoding
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # Create causal mask if needed
        if causal and src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        
        # Transformer forward
        output = self.transformer(x, mask=src_mask)
        output = self.output_norm(output)
        
        return output
