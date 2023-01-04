import torch
from torch import nn

from independent.gpt.model.attention import SelfAttention
from independent.gpt.model.gelu import GELU
from independent.gpt.running.configuration import Configuration


class Block(nn.Module):
    def __init__(self, config: Configuration):
        super().__init__()
        self.config = config

        self.layer_norm_1 = nn.LayerNorm(config.embedding_size)
        self.attention = SelfAttention(config)
        self.layer_norm_2 = nn.LayerNorm(config.embedding_size)

        self.fully_connected = nn.Linear(config.embedding_size, config.fully_connected_size)
        self.activation = GELU()
        self.output_projection = nn.Linear(config.fully_connected_size, config.embedding_size)
        self.dropout = nn.Dropout(config.projection_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input: batch_size x seq_length x embedding_size
        output: batch_size x seq_length x embedding_size
        """

        before_attention = x

        x = self.layer_norm_1(x)
        x = self.attention(x)              # -> batch_size x seq_length x embedding_size
        x = x + before_attention           # residual connection

        before_output_projection = x
        x = self.layer_norm_2(x)
        x = self.fully_connected(x)        # -> batch_size x seq_length x fully_connected_size(e.g. 4X embedding)
        x = self.activation(x)
        x = self.output_projection(x)      # -> batch_size x seq_length x embedding_size
        x = self.dropout(x)
        x = x + before_output_projection   # residual connection

        return x
