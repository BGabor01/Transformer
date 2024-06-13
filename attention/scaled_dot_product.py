from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism

    This mechanism computes the attention weights and the corresponding
    weighted sum of the values
    """

    def __init__(self, temperature: float) -> None:
        """
        Initializes the ScaledDotProductAttention module.

        Args:
            temperature (float): Scaling factor, typically the square root of the key dimension.
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the forward pass for scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_length, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_length, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_length, d_model).
            mask (Optional[torch.Tensor]): Mask tensor of shape (batch_size, seq_length, seq_length), default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and attention weights.
        """
        attention = torch.matmul(query / self.temperature, key.transpose(-2, -1))

        if mask is not None:
            # Ensures that when the softmax function is applied, the corresponding probabilities are effectively zero
            attention = attention.masked_fill(mask == 0, -1e9)

        attention = F.softmax(attention, dim=-1)

        output = torch.matmul(attention, value)

        return output, attention


if __name__ == "__main__":
    import math

    batch_size = 2
    seq_length = 5
    d_model = 8
    temperature = math.sqrt(d_model)

    q = torch.rand(batch_size, seq_length, d_model)
    k = torch.rand(batch_size, seq_length, d_model)
    v = torch.rand(batch_size, seq_length, d_model)

    # All ones in this case, meaning no masking effect
    mask = torch.ones(batch_size, seq_length, seq_length)

    attention = ScaledDotProductAttention(temperature)

    output, attn_weights = attention(q, k, v, mask)

    print("Output:", output)
    print("Attention Weights:", attn_weights)
