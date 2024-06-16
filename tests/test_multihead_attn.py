import pytest
import torch
from model.attention import MultiHeadAttention


class TestMultiHeadAttention:
    @pytest.fixture(autouse=True)
    def setup_attention(self):
        model_dim = 512
        key_dim = 64
        value_dim = 64
        max_len = 500
        n_heads = 8
        self.attention = MultiHeadAttention(
            model_dim, key_dim, value_dim, max_len, n_heads
        )

    def test_forward_pass(self):
        attention = self.attention
        batch_size = 2
        seq_len = 10
        model_dim = 512

        query = torch.rand(batch_size, seq_len, model_dim)
        key = torch.rand(batch_size, seq_len, model_dim)
        value = torch.rand(batch_size, seq_len, model_dim)

        output = attention(query, key, value)
        assert output.shape == (
            batch_size,
            seq_len,
            model_dim,
        ), "Output shape mismatch."

    def test_padding_mask(self):
        attention = self.attention
        batch_size = 2
        seq_len = 10
        model_dim = 512

        query = torch.rand(batch_size, seq_len, model_dim)
        key = torch.rand(batch_size, seq_len, model_dim)
        value = torch.rand(batch_size, seq_len, model_dim)

        pad_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        pad_mask[0, 5:] = 1

        output_with_mask = attention(query, key, value, pad_mask=pad_mask)
        output_without_mask = attention(query, key, value)

        assert not torch.allclose(
            output_with_mask, output_without_mask
        ), "Outputs should be different when using padding mask."

    def test_causal_mask(self):
        model_dim = 512
        key_dim = 64
        value_dim = 64
        max_len = 1000
        n_heads = 8
        attention = MultiHeadAttention(
            model_dim, key_dim, value_dim, max_len, n_heads, casual=True
        )

        batch_size = 2
        seq_len = 10

        query = torch.rand(batch_size, seq_len, model_dim)
        key = torch.rand(batch_size, seq_len, model_dim)
        value = torch.rand(batch_size, seq_len, model_dim)

        output = attention(query, key, value)
        assert output.shape == (
            batch_size,
            seq_len,
            model_dim,
        ), "Output shape mismatch with causal mask."
