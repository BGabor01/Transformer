from typing import Optional

import torch.nn as nn

from model.embedding import InputEmbedding
from .encoder_block import EncoderBlock


class Encoder(nn.Module):
    """
    Encoder consisting of input embedding, multiple encoder blocks, and layer normalization.
    """

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        key_dim: int,
        value_dim: int,
        max_len: int,
        n_heads: int,
        n_layers: int,
        dropout: Optional[float] = 0.1,
    ) -> None:
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            model_dim (int): Dimensionality of the input and output features.
            key_dim (int): Dimensionality of the key and query vectors.
            value_dim (int): Dimensionality of the value vectors.
            max_len (int): Maximum length of the input sequences.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of encoder layers.
            dropout (Optional[float]): Dropout rate (default is 0.1).
        """
        super().__init__()

        self.input_embedding = InputEmbedding(vocab_size, model_dim, dropout)
        transformer_blocks = [
            EncoderBlock(key_dim, value_dim, model_dim, n_heads, max_len, dropout)
            for _ in range(n_layers)
        ]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.norm_layer = nn.LayerNorm(model_dim)

    def forward(self, input, pad_mask=None):
        """
        Forward pass for the encoder.

        Args:
            input (torch.Tensor): Input tensor of shape ``[batch_size, seq_len]`` containing token indices.
            pad_mask (Optional[torch.Tensor]): Padding mask tensor of shape ``[batch_size, seq_len]`` (default is None).

        Returns:
            torch.Tensor: Output tensor of shape ``[batch_size, seq_len, model_dim]`` containing the encoded representations.
        """
        input = self.input_embedding(input)
        for block in self.transformer_blocks:
            input = block(input, pad_mask)

        input = self.norm_layer(input)
        return input
