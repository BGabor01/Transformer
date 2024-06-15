import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Feed-Forward neural network.

    This is typically a two-layer linear transformation with a ReLU activation in between.
    """

    def __init__(self, model_dim: int, ff_dim: int, dropout: float = 0.1) -> None:
        """
        Initializes the FeedForward module.

        Args:
            model_dim (int): Dimension of the model.
            ff_dim (int): Dimension of the feed-forward layer.
            dropout (float): Dropout rate. Default is 0.1.
        """
        super().__init__()
        self.linear1 = nn.Linear(model_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ff_dim, model_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape `[batch_size, seq_len, model_dim]`.

        Returns:
            torch.Tensor: Output tensor of shape `[batch_size, seq_len, model_dim]`.
        """
        return self.linear2(self.dropout(F.relu(self.linear1(input))))
