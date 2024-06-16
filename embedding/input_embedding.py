from typing import Optional
import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class InputEmbedding(nn.Module):
    """Embedding layer with positional encoding for input sequences."""

    def __init__(
        self, vocab_size: int, embedding_dim: int, dropout: Optional[float] = 0.1
    ) -> None:
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimensionality of the embeddings.
        """
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encode = PositionalEncoding(embedding_dim, dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the input embedding layer. Apply position encoding

        Args:
            input (torch.Tensor): Input tensor of shape ``[batch_size, seq_len]`` containing token indices.

        Returns:
            torch.Tensor: Output tensor of shape ``[batch_size, seq_len, embedding_dim]`` containing the embeddings with positional encoding.
        """
        embeddings = self.embeddings(input)
        embed_with_pos = self.pos_encode(embeddings)
        return embed_with_pos
