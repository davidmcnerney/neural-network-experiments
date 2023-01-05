import math

import torch
from torch import nn

from independent.gpt.model.block import Block
from independent.gpt.running.configuration import Configuration


class GPT(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()

        self.config = config

        # Set up our neural network: embedding, transformer layers, and final projection.
        # Question: Do we get any benefit from using ModuleDict here, as opposed to making each thing
        # just an attribute, e.g. self.token_embedding = ... ?
        self.transformer = nn.ModuleDict({
            "token_embedding": nn.Embedding(config.vocabulary_size, config.embedding_size),
            "position_embedding": nn.Embedding(config.block_size, config.embedding_size),
            "dropout": nn.Dropout(config.embedding_dropout),
            "layers": nn.ModuleList([Block(config) for _ in range(config.count_layers)]),
            "layer_norm": nn.LayerNorm(config.embedding_size),
        })
        self.final_output_projection = nn.Linear(config.embedding_size, config.vocabulary_size, bias=False)

        # Initialize weights for our immediate children (or all?)

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
        x = self.transformer.token_embedding(x)   # -> batch_size x seq_length x embedding_size

        # Add position embedding
        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)  # 1 x seq_length
        x = x + self.transformer.position_embedding(positions)

        # Dropout
        x = self.transformer.dropout(x)   # -> batch_size x seq_length x embedding_size

        # Layers
        for block in self.transformer.layers:
            x = block(x)   # -> shape unchanged

        x = self.transformer.layer_norm(x)

        # Final output projection -> logits for each token in our vocabulary
        x = self.final_output_projection(x)     # -> batch_size x seq_length x vocab_size

        return x

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: tensor of token indices: batch_size x seq_length
            this corresponds to an input body of text
        output: tensor of token indices: batch_size x max_output_tokens
            this corresponds to an output body of text
        """
        return x   # TODO

    def summarize_parameters(self) -> None:
        count_parameters = sum(p.numel() for p in self.parameters())
        print(f"Count parameters: {count_parameters}")

        # for name, parameter in self.named_parameters():
        #     print(f"   {name}")

        print("\n")

    def _initialize_weights(self):
        self.apply(self._initialize_weights_callback)

        # Different initialization for some of our internal output projections. Not sure of the exact rationale here
        standard_deviation = 0.02 / math.sqrt(2 * self.config.count_layers)
        for parameter_name, parameter in self.named_parameters():
            if parameter_name.endswith('.output_projection.weight'):
                torch.nn.init.normal_(parameter, mean=0.0, std=standard_deviation)

    @staticmethod
    def _initialize_weights_callback(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
