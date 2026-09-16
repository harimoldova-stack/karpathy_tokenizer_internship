import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.key = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.query = nn.Linear(
            embedding_size,
            embedding_size
        )

        self.value = nn.Linear(
            embedding_size,
            embedding_size
        )

    def forward(self, x):

        batch_size, sequence_length, embedding_size = x.shape

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        scores = query @ key.transpose(-2, -1)

        scores = scores / (embedding_size ** 0.5)

        # Causal mask
        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention = F.softmax(
            scores,
            dim=-1
        )

        output = attention @ value

        return output


class FeedForward(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                embedding_size,
                embedding_size * 4
            ),

            nn.ReLU(),

            nn.Linear(
                embedding_size * 4,
                embedding_size
            )
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            embedding_size
        )

        self.attention = SelfAttention(
            embedding_size
        )

        self.layer_norm_2 = nn.LayerNorm(
            embedding_size
        )

        self.feed_forward = FeedForward(
            embedding_size
        )

    def forward(self, x):

        # Attention + residual
        x = x + self.attention(
            self.layer_norm_1(x)
        )

        # Feed-forward + residual
        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size=64,
        block_size=128,
        num_layers=2
    ):
        super().__init__()

        self.block_size = block_size

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_size
        )

        # Positional embeddings
        self.position_embedding = nn.Embedding(
            block_size,
            embedding_size
        )

        # Multiple Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embedding_size)
            for _ in range(num_layers)
        ])

        # Final LayerNorm
        self.final_layer_norm = nn.LayerNorm(
            embedding_size
        )

        # Language model head
        self.language_model_head = nn.Linear(
            embedding_size,
            vocab_size
        )

    def forward(self, token_ids):

        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.block_size:
            raise ValueError(
                "Sequence length is larger than block size."
            )

        # Token embeddings
        token_vectors = self.token_embedding(
            token_ids
        )

        # Position embeddings
        positions = torch.arange(
            sequence_length,
            device=token_ids.device
        )

        position_vectors = self.position_embedding(
            positions
        )

        # Combine embeddings
        x = token_vectors + position_vectors

        # Pass through Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization
        x = self.final_layer_norm(x)

        # Vocabulary predictions
        logits = self.language_model_head(x)

        return logits