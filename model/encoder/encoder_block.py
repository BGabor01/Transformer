from typing import Optional

import torch
import torch.nn as nn

from model.attention import MultiHeadAttention


class EncoderBlock(nn.Module):
    """
    Encoder block with multi-head attention and feedforward network.
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        model_dim: int,
        n_heads: int,
        max_len: int,
        dropout: Optional[float] = 0.1,
    ) -> None:
        """
        Args:
            key_dim (int): Dimensionality of the key and query vectors.
            value_dim (int): Dimensionality of the value vectors.
            model_dim (int): Dimensionality of the input and output features.
            n_heads (int): Number of attention heads.
            max_len (int): Maximum length of the input sequences.
            dropout (Optional[float]): Dropout rate (default is 0.1).
        """
        super().__init__()
        self.norm_layer1 = nn.LayerNorm(model_dim)
        self.norm_layer2 = nn.LayerNorm(model_dim)
        self.multihead_attn = MultiHeadAttention(
            model_dim, key_dim, value_dim, max_len, n_heads, casual=False
        )
        self.ff_network = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout),
        )
        self.droput = nn.Dropout(dropout)

    def forward(
        self, input: torch.Tensor, pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the encoder block.

        Args:
            input (torch.Tensor): Input tensor of shape ``[batch_size, seq_len, model_dim]``.
            pad_mask (Optional[torch.Tensor]): Padding mask tensor of shape ``[batch_size, seq_len]`` (default is None).

        Returns:
            torch.Tensor: Output tensor of shape ``[batch_size, seq_len, model_dim]``.
        """
        input = self.norm_layer1(
            input + self.multihead_attn(input, input, input, pad_mask)
        )
        input = self.norm_layer2(input + self.ff_network(input))
        input = self.droput(input)
        return input
