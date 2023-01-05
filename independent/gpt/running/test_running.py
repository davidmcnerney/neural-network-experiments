import torch

from independent.gpt.model.gpt import GPT
from independent.gpt.running.configuration import Configuration


def test_forward():
    config = Configuration.nano()
    gpt = GPT(config=config)

    batch_size = 1
    sequence_length = 4
    x = torch.tensor([[100, 200, 300, 400]])   # token indices: batch size x seq length
    assert x.shape == torch.Size([batch_size, sequence_length])
    y = gpt(x)

    assert y.shape == torch.Size([batch_size, sequence_length, config.vocabulary_size])

    print("\n")
    gpt.summarize_parameters()


def test_generate():
    config = Configuration.nano()
    gpt = GPT(config=config)

    batch_size = 1
    sequence_length = 4
    x = torch.tensor([[100, 200, 300, 400]])
    assert x.shape == torch.Size([batch_size, sequence_length])
    max_output_tokens = 3
    y = gpt.generate(x, max_output_tokens=max_output_tokens)

    assert y.shape == torch.Size([batch_size, sequence_length + max_output_tokens])

    print("\n")
    print(y)
