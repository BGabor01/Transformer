from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism.

    This mechanism computes the attention weights and the corresponding
    weighted sum of the values.
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
            query (torch.Tensor): Query tensor of shape `[batch_size, seq_len, d_model]`.
            key (torch.Tensor): Key tensor of shape `[batch_size, seq_len, d_model]`.
            value (torch.Tensor): Value tensor of shape `[batch_size, seq_len, d_model]`.
            mask (Optional[torch.Tensor]): Mask tensor of shape `[batch_size, seq_len, seq_len]`, default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor of shape `[batch_size, seq_len, d_model]` and attention weights of shape `[batch_size, seq_len, seq_len]`.
        """
        attention = torch.matmul(query / self.temperature, key.transpose(-2, -1))

        if mask is not None:
            # Ensures that when the softmax function is applied, the corresponding probabilities are effectively zero
            attention = attention.masked_fill(mask == 0, -1e9)

        attention = F.softmax(attention, dim=-1)

        output = torch.matmul(attention, value)

        return output, attention
