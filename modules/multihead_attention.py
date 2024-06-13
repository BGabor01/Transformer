import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, model_dim: int) -> None:
        super().__init__()
        assert model_dim % n_heads == 0

        self.n_heads = n_heads
        self.key_vector_dim = model_dim // n_heads
        self.value_vector_dim = model_dim // n_heads
        self.model_dim = model_dim

        self.weight_qs = nn.Linear(model_dim, n_heads * self.key_vector_dim)
        self.weight_ks = nn.Linear(model_dim, n_heads * self.key_vector_dim)
        self.weight_vs = nn.Linear(model_dim, n_heads * self.value_vector_dim)
