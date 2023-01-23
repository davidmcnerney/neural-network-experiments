import math

import torch
from torch import nn
from torch.nn import functional as F

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

        # Weight tying - explained here: https://paperswithcode.com/method/weight-tying
        self.transformer.token_embedding.weight = self.final_output_projection.weight

        # TODO: Different initialization for attention head and transformer block output projections

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
        x = self.transformer.token_embedding(x)     # -> batch_size x seq_length x embedding_size

        # Add position embedding
        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)  # 1 x seq_length
        position_embedding = self.transformer.position_embedding(positions)  # 1 x seq_length x embedding_size
        x = x + position_embedding                  # -> batch_size x seq_length x embedding_size (broadcasts 1st dim)

        # Dropout
        x = self.transformer.dropout(x)             # -> batch_size x seq_length x embedding_size

        # Layers
        for block in self.transformer.layers:
            x = block(x)                            # -> shape unchanged

        x = self.transformer.layer_norm(x)

        # Final output projection -> logits for each token in our vocabulary
        x = self.final_output_projection(x)         # -> batch_size x seq_length x vocab_size

        return x

    @staticmethod
    def calculate_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        input:
            logits tensor, as returned from .forward()
            targets tensor: batch size x seq_length, contains labelled token indices
        """
        # We flatten leading dimensions of both logits and targets, so that F.cross_entropy()
        # can match vectors of vocab_size in logits to scalars of token index in targets.
        flattened_logits = logits.view(-1, logits.size(-1))     # (batch_size*seq_length) x vocab_size
        flattened_targets = targets.view(-1)                    # (batch_size*seq_length)
        return F.cross_entropy(flattened_logits, flattened_targets, ignore_index=-1)   # do I need ignore_index -1 here?

    def generate(self, x: torch.Tensor, max_output_tokens: int) -> torch.Tensor:
        """
        input: token indices - batch_size x seq_length
            this corresponds to input bodies of text
        output: tensor of token indices: batch_size x seq_length+max_output_tokens
            this corresponds to input bodies of text, completed

        TODO: support temperature, top_k
        """
        for _ in range (max_output_tokens):
            x_cropped = self._trim_sequence_to_block_size(x)
            next_indices = self.sample(x_cropped)
            x = torch.cat((x, next_indices), dim=1)                 # batch_size x seq_length++
        return x

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: token indices: batch_size x seq_length
            (seq_length must be <= block_size)
        output: tensor of token indices: batch size x 1

        Returns the predicted next token in each sequence of the batch.
        """
        logits = self(x)                                            # batch_size x seq_length x vocab_size
        last_logits = logits[:, -1, :]                              # batch_size x vocab_size
        probs = F.softmax(last_logits, dim=-1)                      # batch_size x vocab_size  (dim=-1 normalizes values in the last dimension, i.e. vocab_size)
        next_indices = torch.multinomial(probs, num_samples=1)      # batch_size x 1
        return next_indices

    def _trim_sequence_to_block_size(self, x: torch.Tensor) -> torch.Tensor:
        """
        Drops older elements of each sequence in the batch so as to keep sequence length below
        configured block_size.

        input: token indices - batch_size x seq length
        output: token indices - batch_size x seq length (clipped to block_size)
        """
        if x.size(1) <= self.config.block_size:
            return x
        else:
            return x[:, -self.config.block_size:]

    def summarize_parameters(self) -> None:
        count_parameters = sum(p.numel() for p in self.parameters())
        print(f"Count parameters: {count_parameters}")

        # for name, parameter in self.named_parameters():
        #     print(f"   {name}")
        # print("\n")

    def _initialize_weights(self):
        self.apply(self._initialize_weights_callback)

        # Different initialization for some of our internal output projections. Not sure of the exact rationale here
        standard_deviation = 0.02 / math.sqrt(2 * self.config.count_layers)
        for parameter_name, parameter in self.named_parameters():
            if parameter_name.endswith('.output_projection.weight'):
                torch.nn.init.normal_(parameter, mean=0.0, std=standard_deviation)

    @staticmethod
    def _initialize_weights_callback(module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
