from typing import Optional
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding class for adding position information to input embeddings.

    The positional encodings are added to the input embeddings
    to provide information about the position of each token in the sequence.
    """

    def __init__(
        self,
        embedding_dim: int,
        dropout: Optional[float] = 0.1,
        max_len: Optional[int] = 2000,
    ) -> None:
        """
        Initializes the PositionalEncoding module.

        Args:
            embedding_dim (int): Dimension of the embedding vectors.
            dropout (Optional[float]): Dropout rate to be applied after adding positional encodings. Default is 0.1.
            max_len (Optional[int]): Maximum length of the sequences. Default is 2000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)

        # Scale the positions with the embedding dim
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )
        pe = torch.zeros(max_len, 1, embedding_dim)

        # Sinusoidal pos encode
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the positional encoding layer.

        Args:
            input (torch.Tensor): Input tensor of shape ``[seq_len, batch_size, embedding_dim]``.

        Returns:
            torch.Tensor: Output tensor with positional encodings added, of shape ``[seq_len, batch_size, embedding_dim]``.
        """
        input = input + self.pe[: input.size(0)]
        return self.dropout(input)
