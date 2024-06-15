from typing import Optional
import torch
import torch.nn as nn
from .decoder_layer import DecoderLayer
from embedding import InputEmbedding


class Decoder(nn.Module):

    def __init__(
        self,
        model_dim: int,
        vocab_size: int,
        n_heads: int,
        n_layer: int,
        ff_dim: int,
        dropout: Optional[float] = 0.1,
    ) -> None:
        super().__init__()
        self.target_embedding = InputEmbedding(vocab_size, model_dim)
        self.layers = nn.ModuleList(
            [DecoderLayer(n_heads, model_dim, ff_dim, dropout) for _ in range(n_layer)]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        encoder_output: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        taget = self.target_embedding(target)

        for layer in self.layers:
            target = layer(target, encoder_output, target_mask, encoder_mask)

        return self.norm(target)
