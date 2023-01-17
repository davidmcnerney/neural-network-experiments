import torch

from independent.gpt.model.gpt import GPT
from independent.gpt.projects.tiny_shakespeare.configuration import configuration
from independent.gpt.projects.tiny_shakespeare.dataset import TinyShakespeareDataset
from independent.gpt.projects.tiny_shakespeare.test_fixtures.fixture_loader import fixture_loader
from independent.gpt.running.training import train


def test_dataset():
    config = configuration()
    config.block_size = 3
    dataset = TinyShakespeareDataset(
        filename=fixture_loader.path_to_file("corpus.txt"),
        vocab_filename=fixture_loader.path_to_file("vocab.json"),
        merge_filename=fixture_loader.path_to_file("merges.txt"),
        block_size=config.block_size,
    )
    assert len(dataset) == 1
    x, y = dataset[0]
    assert torch.equal(x, torch.tensor([684, 1244, 58]))
    assert torch.equal(y, torch.tensor([1244, 58, 10]))


def test_dataset_and_train():
    config = configuration()
    config.block_size = 3
    model = GPT(config=config)
    dataset = TinyShakespeareDataset(
        filename=fixture_loader.path_to_file("corpus.txt"),
        vocab_filename=fixture_loader.path_to_file("vocab.json"),
        merge_filename=fixture_loader.path_to_file("merges.txt"),
        block_size=config.block_size,
    )
    train(model=model, dataset=dataset)
