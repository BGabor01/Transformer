from typing import Optional

import torch
import torch.nn as nn

from .scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    This class implements the multi-head attention mechanism, which allows the model
    to jointly attend to information from different representation subspaces.
    """

    def __init__(self, n_heads: int, model_dim: int) -> None:
        """
        Initializes the MultiHeadAttention module.

        Args:
            n_heads (int): Number of attention heads.
            model_dim (int): Dimension of the model.
        """
        super().__init__()
        assert (
            model_dim % n_heads == 0
        ), "Model dimension must be divisible by the number of heads."

        self.n_heads = n_heads
        self.key_vector_dim = model_dim // n_heads
        self.value_vector_dim = model_dim // n_heads
        self.model_dim = model_dim

        self.weight_qs = nn.Linear(model_dim, n_heads * self.key_vector_dim)
        self.weight_ks = nn.Linear(model_dim, n_heads * self.key_vector_dim)
        self.weight_vs = nn.Linear(model_dim, n_heads * self.value_vector_dim)
        self.output_projection = nn.Linear(n_heads * self.value_vector_dim, model_dim)

        self.attention = ScaledDotProductAttention(temperature=self.key_vector_dim**0.5)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for multi-head attention.

        Args:
            query (torch.Tensor): Query tensor of shape `[batch_size, seq_len, model_dim]`.
            key (torch.Tensor): Key tensor of shape `[batch_size, seq_len, model_dim]`.
            value (torch.Tensor): Value tensor of shape `[batch_size, seq_len, model_dim]`.
            mask (Optional[torch.Tensor]): Mask tensor of shape `[batch_size, seq_len, seq_len]`. Default is None.

        Returns:
            torch.Tensor: Output tensor after applying multi-head attention.
        """
        batch_size = query.size(0)
        query = (
            self.weight_qs(query)
            .view(batch_size, -1, self.n_heads, self.key_vector_dim)
            .transpose(1, 2)
        )

        key = (
            self.weight_ks(key)
            .view(batch_size, -1, self.n_heads, self.key_vector_dim)
            .transpose(1, 2)
        )

        value = (
            self.weight_ks(value)
            .view(batch_size, -1, self.n_heads, self.value_vector_dim)
            .transpose(1, 2)
        )

        output, _ = self.attention(query, key, value, mask)
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.value_vector_dim)
        )
        output = self.output_projection(output)
        return output
