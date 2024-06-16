from typing import Optional

import torch
import torch.nn as nn

from attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    """
    Dencoder block with multi-head attention and feedforward network.
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        model_dim: int,
        n_heads: int,
        max_len: int,
        dropout: Optional[float] = 0.1,
    ):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            model_dim (int): Dimension of the model.
            n_layers (int): Number of decoder layers.
            n_heads (int): Number of attention heads.
            ff_dim (int): Dimension of the feed-forward layer.
            dropout (Optional[float]): Dropout rate. Default is 0.1.
        """
        super().__init__()
        self.norm_layer = nn.LayerNorm(model_dim)
        self.norm_layer2 = nn.LayerNorm(model_dim)
        self.norm_layer3 = nn.LayerNorm(model_dim)
        self.multihead_attn1 = MultiHeadAttention(
            model_dim, key_dim, value_dim, max_len, n_heads, casual=True
        )
        self.multihead_attn2 = MultiHeadAttention(
            model_dim, key_dim, value_dim, max_len, n_heads, casual=False
        )
        self.ff_network = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout),
        )
        self.droput = nn.Dropout(dropout)

    def forward(
        self, encoder_output, decoder_input, encoder_mask=None, decoder_mask=None
    ) -> torch.Tensor:
        """
        Forward pass for the decoder block.

        Args:
            encoder_output (torch.Tensor): Output tensor from the encoder of shape ``[batch_size, seq_len, model_dim]``.
            decoder_input (torch.Tensor): Input tensor for the decoder of shape ``[batch_size, seq_len, model_dim]`` containing embeddings.
            encoder_mask (Optional[torch.Tensor]): Padding mask for the encoder output of shape ``[batch_size, seq_len]`` (default is None).
            decoder_mask (Optional[torch.Tensor]): Padding mask for the decoder input of shape ``[batch_size, seq_len]`` (default is None).

        Returns:
            torch.Tensor: Output tensor of shape ``[batch_size, seq_len, model_dim]`` containing the processed representations.
        """
        output = self.norm_layer(
            decoder_input
            + self.multihead_attn1(
                decoder_input, decoder_input, decoder_input, decoder_mask
            )
        )

        output = self.norm_layer2(
            output
            + self.multihead_attn2(output, encoder_output, encoder_output, encoder_mask)
        )

        output = self.norm_layer3(output + self.ff_network(output))

        return output
