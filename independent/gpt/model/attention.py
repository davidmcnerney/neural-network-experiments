import math

import torch
from torch import nn
from torch.nn import functional as F

from independent.gpt.running.configuration import Configuration


class SelfAttention(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self._validate_configuration(config)
        self.config = config

        # Query, key, and value projections, combined together and for all samples in the batch
        #   queries and keys are conceptually symmetrical and generate self-attention matrix
        self.query_key_value_projection = nn.Linear(config.embedding_size, 3 * config.embedding_size)

        # Mask - we use this to prevent attention from earlier tokens to later ones in the sequence
        # 1 x 1 x block_size x block_size
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

        self.attention_dropout = nn.Dropout(config.attention_dropout)

        self.output_projection = nn.Linear(config.embedding_size, config.embedding_size)

        self.output_dropout = nn.Dropout(config.projection_dropout)

    @staticmethod
    def _validate_configuration(config):
        if config.embedding_size % config.count_heads != 0:
            raise Exception("Uneven number of heads")
        if config.embedding_size // config.count_heads != config.head_size:
            raise Exception("Incorrect head size, given other config")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: batch_size x seq_length x embedding_size
        output: batch_size x seq_length x embedding_size
        """

        self._validate_input(x)

        batch_size = x.size(0)
        sequence_length = x.size(1)

        # Use our linear projection to compute query, key, and value tensors together.
        #   -> batch_size x sequence_length x embedding_size
        query, key, value = self.query_key_value_projection(x).split(self.config.embedding_size, dim=2)

        # Split into separate heads, and then transpose so that the two final dimensions are sequence_length x head_size
        #   -> batch_size x count_heads x sequence_length x head_size
        query = query.view(batch_size, sequence_length, self.config.count_heads, self.config.head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.config.count_heads, self.config.head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.config.count_heads, self.config.head_size).transpose(1, 2)

        # Attention matrix
        #   -> batch_size x count_heads x sequence_length x sequence_length
        attention = query @ key.transpose(-2, -1)
        attention /= math.sqrt(self.config.head_size)

        # Masking
        #   Note that our mask tensor is sized to block_size, which is the max sequence length. Our current
        #   sequence length may be shorter, so we have to trim the last two dimensions of the mask before use.
        #   -> shape unchanged
        attention = attention.masked_fill(self.mask[:, :, :sequence_length, :sequence_length] == 0, float("-inf"))

        # Softmax dim=-1, which processes each row (dim=-2) such that its elements sum to 1.0
        #   -> shape unchanged
        attention = F.softmax(attention, dim=-1)

        attention = self.attention_dropout(attention)

        # Value multiplication
        # (batch_size x count_heads x sequence_length x sequence_length) @ (batch_size x count_heads x sequence_length x head_size)
        #   -> batch_size x count_heads x sequence_length x head_size
        y = attention @ value

        # Re-assemble
        # We swap count_heads and sequence_length dimensions (1 and 2) again
        #   -> batch_size x sequence_length x count_heads x head_size
        y = y.transpose(1, 2)
        # And then use .view() to concat the head outputs together
        #   -> batch_size x sequence_length x embedding_size
        y = y.contiguous().view(batch_size, sequence_length, self.config.embedding_size)

        # Output projection
        #   -> batch_size x sequence_length x embedding_size
        y = self.output_projection(y)

        y = self.output_dropout(y)

        return y

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise Exception("Input does not have 3 dimensions")
        if x.size(1) > self.config.block_size:
            raise Exception("Input sequence length > configured block size")
        if x.size(2) != self.config.embedding_size:
            raise Exception("Input embedding size does not match configured")
