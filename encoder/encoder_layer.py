from typing import Optional

import torch
import torch.nn as nn

from attention import MultiHeadAttention
from .feed_forward import FeedForward


class EncoderLayer(nn.Module):
    """
    Encoder layer consisting of multi-head attention and feed-forward neural network.
    """

    def __init__(
        self, model_dim: int, n_head: int, ff_dim: int, dropout: Optional[float] = 0.1
    ) -> None:
        """
        Initializes the EncoderLayer module.

        Args:
            model_dim (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward layer.
            dropout (float): Dropout rate. Default is 0.1.
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head, model_dim)
        self.feed_forward = FeedForward(model_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, input_mask: Optional[torch.Tensor] = None):
        """
        Forward pass for the encoder layer.

        Args:
            input (torch.Tensor): Input tensor of shape `[batch_size, seq_len, model_dim]`.
            input_mask (Optional[torch.Tensor]): Mask tensor of shape `[batch_size, seq_len, seq_len]`. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, seq_len, model_dim]`.
        """

        attn_output = self.self_attn(input, input, input, input_mask)
        input = input + self.dropout(attn_output)
        input = input + self.norm1(input)

        ff_output = self.feed_forward(input)
        input = input + self.dropout(ff_output)
        input = input + self.norm2(input)

        return input
