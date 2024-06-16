import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


class SeqToSeqTransformer(nn.Module):
    """
    SeqToSeqTransformer model consisting of an encoder and a decoder.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
    ):
        """
        Forward pass of the SeqToSeqTransformer.

        Args:
            encoder_input (torch.Tensor): Input tensor to the encoder of shape ``[batch_size, seq_len]``.
            decoder_input (torch.Tensor): Input tensor to the decoder of shape ``[batch_size, seq_len]``.
            encoder_mask (torch.Tensor): Mask tensor for the encoder input of shape ``[batch_size, seq_len]``.
            decoder_mask (torch.Tensor): Mask tensor for the decoder input of shape ``[batch_size, seq_len]``.

        Returns:
            torch.Tensor: Output tensor from the decoder of shape [batch_size, seq_len, model_dim].
        """
        encoder_output = self.encoder(encoder_input, encoder_mask)
        decoder_output = self.decoder(
            encoder_output, decoder_input, encoder_mask, decoder_mask
        )

        return decoder_output
