from typing import Optional

import torch.nn as nn

from embedding import InputEmbedding
from .decoder_block import DecoderBlock


class Decoder(nn.Module):
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
        Decoder consisting of input embedding, multiple decoder blocks, layer normalization, and output layer.
        """
        super().__init__()
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            model_dim (int): Dimensionality of the input and output features.
            key_dim (int): Dimensionality of the key and query vectors.
            value_dim (int): Dimensionality of the value vectors.
            max_len (int): Maximum length of the input sequences.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of decoder layers.
            dropout (Optional[float]): Dropout rate (default is 0.1).
        """
        self.input_embedding = InputEmbedding(vocab_size, model_dim, dropout)
        transformer_blocks = [
            DecoderBlock(key_dim, value_dim, model_dim, n_heads, max_len, dropout)
            for _ in range(n_layers)
        ]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)
        self.norm_layer = nn.LayerNorm(model_dim)
        self.output_layer = nn.Linear(model_dim, vocab_size)

    def forward(
        self, encoder_output, decoder_input, encoder_mask=None, decoder_mask=None
    ):
        """
        Forward pass for the decoder.

        Args:
            encoder_output (torch.Tensor): Output tensor from the encoder of shape ``[batch_size, seq_len, model_dim]``.
            decoder_input (torch.Tensor): Input tensor for the decoder of shape ``[batch_size, seq_len]`` containing token indices.
            encoder_mask (Optional[torch.Tensor]): Padding mask for the encoder output of shape ``[batch_size, seq_len]`` (default is None).
            decoder_mask (Optional[torch.Tensor]): Padding mask for the decoder input of shape ``[batch_size, seq_len]`` (default is None).

        Returns:
            torch.Tensor: Output tensor of shape ``[batch_size, seq_len, vocab_size]`` containing the decoded representations.
        """
        output = self.input_embedding(decoder_input)
        for block in self.transformer_blocks:
            output = block(encoder_output, output, encoder_mask, decoder_mask)

        output = self.norm_layer(output)
        # many-to-many
        output = self.output_layer(output)
        return output
