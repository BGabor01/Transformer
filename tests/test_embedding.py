import pytest
import torch

from model.embedding import InputEmbedding


class TestEnbedding:

    @pytest.fixture
    def embedding(self):
        vocab_size = 100
        embed_dim = 50
        return InputEmbedding(vocab_size, embed_dim)

    @pytest.mark.smoke
    def test_embedding_values(self, embedding):
        input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
        output_tensor = embedding(input_tensor)

        expected_shape = (2, 3, embedding.embeddings.embedding_dim)
        assert (
            output_tensor.shape == expected_shape
        ), f"Expected shape {expected_shape}, but got {output_tensor.shape}"

    @pytest.mark.slow
    def test_embedding_grad(self, embedding):
        input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        output_tensor = embedding(input_tensor)
        output_tensor.mean().backward()
        assert (
            embedding.embeddings.weight.grad is not None
        ), "Gradients are not computed"


if __name__ == "__main__":
    pytest.main()
