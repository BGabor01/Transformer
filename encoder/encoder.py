from typing import Optional

import torch
import torch.nn as nn

from embedding import InputEmbedding
from transformer import TransformerBlock


class Encoder(nn.Module):
    """
    Encoder consisting of multiple Transformer blocks.

    This module encodes the input sequence into a continuous representation that
    the decoder can use to generate the output sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: Optional[float] = 0.1,
    ):
        """
        Initializes the Encoder module.

        Args:
            vocab_size (int): Size of the vocabulary.
            model_dim (int): Dimension of the model.
            n_layers (int): Number of encoder layers.
            n_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward layer.
            dropout (Optional[float]): Dropout rate. Default is 0.1.
        """
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size, model_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(model_dim, n_heads, ff_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self, input: torch.Tensor, input_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the encoder.

        Args:
            input (torch.Tensor): Input tensor of shape `[batch_size, seq_len]`.
            input_mask (Optional[torch.Tensor]): Mask tensor of shape `[batch_size, 1, 1, seq_len]`. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, seq_len, model_dim]`.
        """
        input = self.input_embedding(input)
        for layer in self.transformer_blocks:
            input = layer(input, input_mask)
        return self.norm(input)
