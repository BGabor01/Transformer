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
        super().__init__()
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
        output = self.input_embedding(decoder_input)
        for block in self.transformer_blocks:
            output = block(encoder_output, output, encoder_mask, decoder_mask)

        output = self.norm_layer(output)
        # many-to-many
        output = self.output_layer(output)
        return output
