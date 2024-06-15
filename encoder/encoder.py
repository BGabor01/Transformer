from typing import Optional
import torch
import torch.nn as nn

from embedding import InputEmbedding
from .encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """
    Encoder consisting of multiple encoder layers.
    """

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: Optional[float] = 0.01,
    ) -> None:
        """
        Initializes the Encoder module.

        Args:
            vocab_size (int): Size of the vocabulary.
            model_dim (int): Dimension of the model.
            n_layers (int): Number of encoder layers.
            n_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward layer.
            dropout (float): Dropout rate. Default is 0.1.
        """
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size, model_dim)
        self.layers = nn.ModuleList(
            [EncoderLayer(model_dim, n_heads, ff_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self, input: torch, input_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for the encoder.

        Args:
            src (torch.Tensor): Input tensor of shape `[batch_size, seq_len]`.
            src_mask (Optional[torch.Tensor]): Mask tensor of shape `[batch_size, seq_len, seq_len]`. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, seq_len, model_dim]`.
        """
        input = self.input_embedding(input)

        for layer in self.layers:
            input = layer(input, input_mask)

        return self.norm(input)
