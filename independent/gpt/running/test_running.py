import torch
import torch.utils.data

from independent.gpt.model.gpt import GPT
from independent.gpt.running.configuration import Configuration
from independent.gpt.running.training import train


def test_train():
    config = Configuration.nano()
    config.vocabulary_size = 4
    config.count_epochs = 1
    config.batch_size = 4
    model = GPT(config=config)
    dataset = torch.utils.data.TensorDataset(
        torch.Tensor([                  # x
            [100, 200, 300, 400],
            [101, 201, 301, 401],
        ]),
        torch.Tensor([                  # y
            [10, 20, 30, 40],
            [11, 21, 31, 41],
        ]),
    )
    train(model=model, dataset=dataset)


def test_forward():
    config = Configuration.nano()
    model = GPT(config=config)

    batch_size = 1
    sequence_length = 4
    x = torch.tensor([[100, 200, 300, 400]])   # token indices: batch size x seq length
    assert x.shape == torch.Size([batch_size, sequence_length])
    y = model(x)

    assert y.shape == torch.Size([batch_size, sequence_length, config.vocabulary_size])

    print("\n")
    model.summarize_parameters()


def test_generate():
    config = Configuration.nano()
    model = GPT(config=config)

    batch_size = 1
    sequence_length = 4
    x = torch.tensor([[100, 200, 300, 400]])
    assert x.shape == torch.Size([batch_size, sequence_length])
    max_output_tokens = 3
    y = model.generate(x, max_output_tokens=max_output_tokens)

    assert y.shape == torch.Size([batch_size, sequence_length + max_output_tokens])

    print("\n")
    print(y)

