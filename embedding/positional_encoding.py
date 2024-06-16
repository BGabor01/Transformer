import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer that adds positional information to the input embeddings
    """

    def __init__(
        self,
        embedding_dim: int,
        dropout: Optional[float] = 0.1,
        max_len: Optional[int] = 2048,
    ) -> None:
        """
        Args:
            embedding_dim (int): Dimensionality of the embeddings.
            dropout (Optional[float]): Dropout rate (default is 0.1).
            max_len (Optional[int]): Maximum length of the input sequences (default is 2048).
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        # Scalling values
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(1, max_len, embedding_dim)

        # Sinusoidal encoding
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the positional encoding layer.

        Args:
            input (torch.Tensor): Input tensor of shape ``[batch_size, seq_len, embedding_dim]``.

        Returns:
            torch.Tensor: Output tensor of shape ``[batch_size, seq_len, embedding_dim]`` with added positional encoding.
        """
        input = input + self.pe[:, : input.size(1), :]
        return self.dropout(input)
