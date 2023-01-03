import torch
from torch import nn

from independent.gpt.running.configuration import Configuration


class GPT(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()

        self.config = config

        # Set up our neural network: embedding, transformer layers, and final projection.
        self.transformer = nn.ModuleDict({
            "token_embedding": nn.Embedding(config.vocabulary_size, config.embedding_size),
            "position_embedding": nn.Embedding(config.block_size, config.embedding_size),
            "dropout": nn.Dropout(config.embedding_dropout),
            # TODO: one Block per layer, contained in a nn.ModuleList
            # TODO: LayerNorm
            # TODO: final projection Linear
        })

        # Initialize weights for our immediate children (or all?)
        self.apply(self._initialize_weights)

        # Different initialization for attention head and transformer block output projections
        # Andrej called these "residual" projections, but I don't yet understand why
        # TODO

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: tensor of token indices: batch_size x seq_length
            this corresponds to an input body of text
        output: tensor of logits: batch_size x seq_length x vocab_size
            this corresponds to a single next token of output text

        seq_length is limited by config.block_size.
        """

        # Input checks
        if x.size(1) > self.config.block_size:
            raise Exception("Input sequence length exceeds block size")

        # Token embedding
        x = self.transformer.token_embedding(x)   # batch_size x seq_length x embedding_size

        # Add position embedding
        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)  # 1 x seq_length
        x = x + self.transformer.position_embedding(positions)

        # Dropout
        x = self.transformer.dropout(x)

        # TODO: all the layers

        # TODO: layer norm

        # TODO: final projection

        return x

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: tensor of token indices: batch_size x seq_length
            this corresponds to an input body of text
        output: tensor of token indices: batch_size x max_output_tokens
            this corresponds to an output body of text
        """
        return x   # TODO

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
