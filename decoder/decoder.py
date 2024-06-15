from typing import Optional
import torch
import torch.nn as nn
from .decoder_layer import DecoderLayer
from embedding import InputEmbedding


class Decoder(nn.Module):
    """Decoder consisting of multiple decoder layers"""

    def __init__(
        self,
        model_dim: int,
        vocab_size: int,
        n_heads: int,
        n_layer: int,
        ff_dim: int,
        dropout: Optional[float] = 0.1,
    ) -> None:
        """
        Initializes the Encoder module.

        Args:
        model_dim (int): Dimension of the model.
        vocab_size (int): Size of the vocabulary.
        n_heads (int): Number of attention heads.
        n_layer (int): Number of decoder layers.
        ff_dim (int): Dimension of the feed-forward layer.
        dropout (Optional[float]): Dropout rate. Default is 0.1.
        """
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
        """
        Forward pass for the decoder.

        Args:
            target (torch.Tensor): Target tensor of shape `[batch_size, seq_len]`.
            encoder_output (torch.Tensor): Encoder output tensor of shape `[batch_size, seq_len, model_dim]`.
            target_mask (Optional[torch.Tensor]): Target mask tensor of shape `[batch_size, 1, target_len, target_len]`. Default is None.
            encoder_mask (Optional[torch.Tensor]): Encoder mask tensor of shape `[batch_size, 1, 1, mem_len]`. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, seq_len, model_dim]`.
        """

        target = self.target_embedding(target)

        for layer in self.layers:
            target = layer(target, encoder_output, target_mask, encoder_mask)

        return self.norm(target)
