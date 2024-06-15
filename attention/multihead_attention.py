from typing import Optional

import torch
import torch.nn as nn

from scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, n_heads: int, model_dim: int) -> None:
        """
        Args:
        n_heads (int): Number of attention heads.
        key_vector_dim (int): Dimension of the key vectors for each head.
        value_vector_dim (int): Dimension of the value vectors for each head.
        model_dim (int): Dimension of the model.
        weight_qs (nn.Linear): Linear layer to project the query vectors.
        weight_ks (nn.Linear): Linear layer to project the key vectors.
        weight_vs (nn.Linear): Linear layer to project the value vectors.
        output_projection (nn.Linear): Linear layer for the final output projection.
        attention (ScaledDotProductAttention): Scaled dot-product attention mechanism.
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
            query (torch.Tensor): Query tensor of shape (batch_size, seq_length, model_dim).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_length, model_dim).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_length, model_dim).
            mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, seq_length, seq_length). Default is None.

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


if __name__ == "__main__":
    batch_size = 2
    seq_length = 5
    model_dim = 8
    n_heads = 2

    q = torch.rand(batch_size, seq_length, model_dim)
    k = torch.rand(batch_size, seq_length, model_dim)
    v = torch.rand(batch_size, seq_length, model_dim)

    mask = torch.ones(batch_size, seq_length, seq_length)

    multihead_attention = MultiHeadAttention(n_heads, model_dim)
    output = multihead_attention(q, k, v, mask)
    print(output)
