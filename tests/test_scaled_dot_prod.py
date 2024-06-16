import pytest
import torch
import torch.nn.functional as F

from model.attention.scaled_dot_product import ScaledDotProductAttention


class TestScaledDotProductAttention:
    @pytest.fixture
    def attention(self):
        return ScaledDotProductAttention(value_dim=64)

    def test_forward_output_shape(self, attention):
        batch_size = 2
        n_heads = 4
        seq_len = 10
        key_dim = 64
        value_dim = 64

        query = torch.randn(batch_size, n_heads, seq_len, key_dim)
        key = torch.randn(batch_size, n_heads, seq_len, key_dim)
        value = torch.randn(batch_size, n_heads, seq_len, value_dim)

        output = attention(query, key, value, query_len=seq_len, value_len=seq_len)
        assert output.shape == (
            batch_size,
            n_heads,
            seq_len,
            value_dim,
        ), f"Expected output shape {(batch_size, n_heads, seq_len, value_dim)}, but got {output.shape}"

    def test_forward_padding_mask(self, attention):
        batch_size = 2
        n_heads = 4
        seq_len = 10
        key_dim = 64
        value_dim = 64

        query = torch.randn(batch_size, n_heads, seq_len, key_dim)
        key = torch.randn(batch_size, n_heads, seq_len, key_dim)
        value = torch.randn(batch_size, n_heads, seq_len, value_dim)
        pad_mask = torch.randint(0, 2, (batch_size, seq_len)).bool()

        output = attention(
            query, key, value, query_len=seq_len, value_len=seq_len, pad_mask=pad_mask
        )
        assert output.shape == (
            batch_size,
            n_heads,
            seq_len,
            value_dim,
        ), f"Expected output shape {(batch_size, n_heads, seq_len, value_dim)}, but got {output.shape}"

    def test_forward_causal_mask(self, attention):
        batch_size = 2
        n_heads = 4
        seq_len = 10
        key_dim = 64
        value_dim = 64

        query = torch.randn(batch_size, n_heads, seq_len, key_dim)
        key = torch.randn(batch_size, n_heads, seq_len, key_dim)
        value = torch.randn(batch_size, n_heads, seq_len, value_dim)
        causal_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len))).bool()

        output = attention(
            query,
            key,
            value,
            query_len=seq_len,
            value_len=seq_len,
            casual=True,
            casual_mask=causal_mask,
        )
        assert output.shape == (
            batch_size,
            n_heads,
            seq_len,
            value_dim,
        ), f"Expected output shape {(batch_size, n_heads, seq_len, value_dim)}, but got {output.shape}"

    def test_forward_with_pad_and_causal_mask(self, attention):
        batch_size = 2
        n_heads = 4
        seq_len = 10
        key_dim = 64
        value_dim = 64

        query = torch.randn(batch_size, n_heads, seq_len, key_dim)
        key = torch.randn(batch_size, n_heads, seq_len, key_dim)
        value = torch.randn(batch_size, n_heads, seq_len, value_dim)
        pad_mask = torch.randint(0, 2, (batch_size, seq_len)).bool()
        causal_mask = torch.tril(torch.ones((1, 1, seq_len, seq_len))).bool()

        output = attention(
            query,
            key,
            value,
            query_len=seq_len,
            value_len=seq_len,
            pad_mask=pad_mask,
            casual=True,
            casual_mask=causal_mask,
        )
        assert output.shape == (
            batch_size,
            n_heads,
            seq_len,
            value_dim,
        ), f"Expected output shape {(batch_size, n_heads, seq_len, value_dim)}, but got {output.shape}"


if __name__ == "__main__":
    pytest.main()
