from typing import Optional

import torch
import torch.nn as nn

from attention import MultiHeadAttention
from neural_network import FeedForward


class DecoderLayer(nn.Module):
    """
    Decoder layer consisting of masked multi-head attention, encoder-decoder multi-head attention,
    and feed-forward neural network.
    """

    def __init__(
        self, n_heads: int, model_dim: int, ff_dim: int, dropout: Optional[float] = 0.01
    ) -> None:
        """
        Initializes the DecoderLayer module.

        Args:
            model_dim (int): Dimension of the model.
            n_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward layer.
            dropout (float): Dropout rate. Default is 0.1.
        """
        super().__init__()
        self.masker_attn = MultiHeadAttention(n_heads, model_dim)
        self.enc_dec_attn = MultiHeadAttention(n_heads, model_dim)
        self.feed_forward = FeedForward(model_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        target: torch.Tensor,
        encoder_output: torch.Tensor,
        target_mask: torch.Tensor,
        encoder_mask,
    ):
        """
        Forward pass for the decoder layer.

        Args:
            target (torch.Tensor): Target tensor of shape `[batch_size, seq_len, model_dim]`.
            encoder_output (torch.Tensor): Encoder output tensor of shape `[batch_size, seq_len, model_dim]`.
            target_mask (torch.Tensor): Target mask tensor of shape `[batch_size, 1, target_len, target_len]`. Default is None.
            encoder_mask (Optional[torch.Tensor]): encoder mask tensor of shape `[batch_size, 1, 1, mem_len]`. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, seq_len, model_dim]`.
        """

        masked_attn = self.masker_attn(target, target, target, target_mask)
        target = target + self.dropout(masked_attn)
        target = target + self.norm1(target)

        enc_dec_attn = self.enc_dec_attn(
            target, encoder_output, encoder_output, encoder_mask
        )
        target = target + self.dropout(enc_dec_attn)
        target = target + self.norm2(target)

        ff_output = self.feed_forward(target)
        target = self.dropout(ff_output)
        target = self.norm3(target)

        return target
