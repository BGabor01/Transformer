from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism
    Calculates the weighted sums of the attention scores.

    Attributes:
        value_dim (int): Dimensionality of the value vectors.
    """

    def __init__(self, value_dim: int):
        super().__init__()
        self.value_dim = value_dim

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_len: int,
        value_len: int,
        pad_mask: Optional[torch.Tensor] = None,
        casual: Optional[bool] = False,
        casual_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the scaled dot-product attention mechanism.

        Args:
            query (torch.Tensor): Query tensor of shape ``[batch_size, n_heads, seq_len, key_dim]``.
            key (torch.Tensor): Key tensor of shape ``[batch_size, n_heads, seq_len, key_dim]``.
            value (torch.Tensor): Value tensor of shape ``[batch_size, n_heads, seq_len, key_dim]``.
            query_len (int): Length of the query sequence.
            value_len (int): Length of the value sequence.
            pad_mask (Optional[torch.Tensor]): Padding mask tensor of shape ``[batch_size, seq_len]`` (default is None).
            causal (Optinal[bool]): Whether to apply a causal mask (default is False).
            causal_mask (Optional[torch.Tensor]): Causal mask tensor of shape ``[1, 1, max_len, max_len]`` (default is None).

        Returns:
            torch.Tensor: Output tensor of shape ``[batch_size, n_heads, seq_len, value_dim]``.
        """
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.value_dim
        )
        if pad_mask is not None:
            # Reshape the mask from [batch_size, seq_len] -> [batch_size, None, None, seq_len]
            attn_scores = attn_scores.masked_fill(
                pad_mask[:, None, None, :] == 0, float("-inf")
            )

        if casual and casual_mask is not None:
            attn_scores = attn_scores.masked_fill(
                casual_mask[:, :, :query_len, :value_len] == 0, float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        weighted_sum = torch.matmul(attn_weights, value)
        return weighted_sum
