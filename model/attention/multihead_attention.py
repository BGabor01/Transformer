from typing import Optional

import torch
import torch.nn as nn

from .scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for seq-to-seq model
    """

    def __init__(
        self,
        model_dim: int,
        key_dim: int,
        value_dim: int,
        max_len: int,
        n_heads: int,
        casual: Optional[bool] = False,
    ) -> None:
        """
        Args:
            model_dim (int): Dimensionality of the input and output features.
            key_dim (int): Dimensionality of the key and query vectors.
            value_dim (int): Dimensionality of the value vectors.
            max_len (int): Maximum length of the input sequences.
            n_heads (int): Number of attention heads.
            causal (Optional[bool]): Whether to apply a causal mask (default is False).
        """
        super().__init__()

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.n_heads = n_heads

        self.query = nn.Linear(model_dim, key_dim * n_heads)
        self.key = nn.Linear(model_dim, key_dim * n_heads)
        self.value = nn.Linear(model_dim, value_dim * n_heads)
        self.final = nn.Linear(value_dim * n_heads, model_dim)
        self.attention = ScaledDotProductAttention(self.value_dim)
        self.casual = casual

        if casual:
            # lower triangle matrix to prevent prevent attending to future positions
            nopeak_mask = torch.tril(torch.ones(max_len, max_len))
            self.register_buffer(
                "nopeak_mask", nopeak_mask.view(1, 1, max_len, max_len)
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the multi-head attention mechanism.

        Args:
            query (torch.Tensor): Query tensor of shape ``[batch_size, seq_len, model_dim]``.
            key (torch.Tensor): Key tensor of shape ``[batch_size, seq_len, model_dim]``.
            value (torch.Tensor): Value tensor of shape ``[batch_size, seq_len, model_dim]``.
            pad_mask (Optional[torch.Tensor]): Padding mask tensor of shape ``[batch_size, seq_len]`` (default is None).

        Returns:
            torch.Tensor: Output tensor of shape ``[batch_size, seq_len, model_dim]``.
        """
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]
        value_len = value.shape[1]

        query = (
            self.query(query)
            .view(batch_size, query_len, self.n_heads, self.key_dim)
            .transpose(1, 2)
        )
        key = (
            self.key(key)
            .view(batch_size, key_len, self.n_heads, self.key_dim)
            .transpose(1, 2)
        )
        value = (
            self.value(value)
            .view(batch_size, value_len, self.n_heads, self.value_dim)
            .transpose(1, 2)
        )

        weighted_sum = self.attention(
            query,
            key,
            value,
            query_len,
            value_len,
            pad_mask,
            self.casual,
            casual_mask=self.nopeak_mask if self.casual else None,
        )

        weighted_sum = (
            weighted_sum.transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, self.value_dim * self.n_heads)
        )

        return self.final(weighted_sum)
