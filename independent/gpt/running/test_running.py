import torch

from independent.gpt.model.gpt import GPT
from independent.gpt.running.configuration import Configuration


def test_forward():
    batch_size = 1
    sequence_length = 4
    x = torch.tensor([[100, 200, 300, 400]])   # token indices: batch size x seq length
    assert x.shape == torch.Size([batch_size, sequence_length])

    config = Configuration.standard()
    gpt = GPT(config=config)
    y = gpt(x)

    assert y.shape == torch.Size([batch_size, sequence_length, config.vocabulary_size])

    print("\n")
    gpt.summarize_parameters()
