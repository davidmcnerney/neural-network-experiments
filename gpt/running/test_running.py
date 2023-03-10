import torch
import torch.utils.data

from gpt.model.gpt import GPT
from gpt.running.configuration import Configuration


def test_forward():
    config = Configuration.for_tests()
    config.batch_size = 1
    model = GPT(config=config)

    sequence_length = 4
    x = torch.tensor([[100, 200, 300, 400]])   # token indices: batch size x seq length
    assert x.shape == torch.Size([config.batch_size, sequence_length])

    y = model(x)
    assert y.shape == torch.Size([config.batch_size, sequence_length, config.vocabulary_size])

    print("\n")
    model.summarize_parameters()


def test_forward_multiple_examples_in_batch():
    config = Configuration.for_tests()
    config.batch_size = 2
    model = GPT(config=config)

    sequence_length = 4
    x = torch.tensor([  # token indices: batch size x seq length
        [100, 200, 300, 400],
        [101, 201, 301, 401],
    ])
    assert x.shape == torch.Size([config.batch_size, sequence_length])

    y = model(x)
    assert y.shape == torch.Size([config.batch_size, sequence_length, config.vocabulary_size])

    print("\n")
    model.summarize_parameters()


def test_generate():
    config = Configuration.for_tests()
    config.batch_size = 1
    model = GPT(config=config)

    sequence_length = 4
    x = torch.tensor([[100, 200, 300, 400]])
    assert x.shape == torch.Size([config.batch_size, sequence_length])
    max_output_tokens = 3

    y = model.generate(x, max_output_tokens=max_output_tokens)
    assert y.shape == torch.Size([config.batch_size, sequence_length + max_output_tokens])

    print("\n")
    print(y)


def test_standard_configurations():
    # This test will surface any basic errors in configuration parameters that result in failed assertions
    # on model instantiation.

    all_configurations = [
        Configuration.for_tests(),
        Configuration.standard(),
    ]
    for config in all_configurations:
        GPT(config=config).summarize_parameters()
