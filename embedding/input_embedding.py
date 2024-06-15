import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class InputEmbedding(nn.Module):
    """
    Input embedding class that combines word embeddings with positional encodings.

    This class creates word embeddings for a given vocabulary size and embedding dimension,
    and then adds positional encodings to these embeddings to provide information about the position
    of each token in the sequence
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        """
        Initializes the InputEmbedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encode = PositionalEncoding(embedding_dim)

    def forward(self, input: torch.Tensor):
        """
        Forward pass for the input embedding layer.

        This method takes input token indices, converts them to embeddings, adds positional encodings,
        and applies dropout.

        Args:
            input (torch.Tensor): Input tensor of shape ``[batch_size, seq_len]``, where batch_size is the number of sequences in the batch,
                                  and seq_len is the sequence length.

        Returns:
            torch.Tensor: Output tensor with word embeddings and positional encodings added, of shape ``[batch_size, seq_len, embedding_dim]``.
        """
        embeddings = self.embeddings(input)
        embed_with_pos = self.pos_encode(embeddings)

        return embed_with_pos


if __name__ == "__main__":

    sentence = "The apple is on the tree"
    tokens = sentence.lower().split()
    vocab = {word: idx for idx, word in enumerate(tokens)}
    indexed_tokens = [vocab[word] for word in tokens]

    print("Tokens:", tokens)
    print("Vocab:", vocab)
    print("Indexed Tokens:", indexed_tokens)

    vocab_size = 100
    embedding_dim = 512
    model = InputEmbedding(vocab_size, embedding_dim)
    input_tokens = torch.tensor([indexed_tokens], dtype=torch.long)

    embeddings = model(input_tokens)

    print("input shape", input_tokens.shape)
    print("Embeddings Shape:", embeddings.shape)
    print("Embeddings:", embeddings)
