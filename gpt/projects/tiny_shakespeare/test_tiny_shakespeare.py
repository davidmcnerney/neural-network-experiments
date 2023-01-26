import torch

from gpt.model.gpt import GPT
from gpt.projects.tiny_shakespeare.configuration import configuration
from gpt.projects.tiny_shakespeare.test_fixtures.fixture_loader import fixture_loader
from gpt.running.dataset import InMemoryTokenDataset
from gpt.running.train import train


# encoded.txt was generated with this command:
# PYTHONPATH=.  pipenv run python gpt/bpe/tokenize_tool.py --input-file gpt/projects/tiny_shakespeare/test_fixtures/corpus.txt --vocab-file gpt/projects/tiny_shakespeare/test_fixtures/vocab.json --merge-file gpt/projects/tiny_shakespeare/test_fixtures/merges.bpe --output-file gpt/projects/tiny_shakespeare/test_fixtures/encoded.txt


def test_dataset():
    training_dataset, validation_dataset = InMemoryTokenDataset.load(
        filename=fixture_loader.path_to_file("corpus.txt"),
        vocab_filename=fixture_loader.path_to_file("vocab.json"),
        merge_filename=fixture_loader.path_to_file("merges.bpe"),
        block_size=8,
        device=torch.device("cpu"),
    )

    assert len(training_dataset) == 280 - 8
    x, y = training_dataset[0]
    assert torch.equal(x, torch.tensor([684, 1244, 58, 10, 2562, 342, 3023, 2598]))  # these are from encoded.txt
    assert torch.equal(y, torch.tensor([1244, 58, 10, 2562, 342, 3023, 2598, 2524]))

    assert len(validation_dataset) == (311 - 280) - 8
    x, y = validation_dataset[0]
    assert torch.equal(x, torch.tensor([2440, 623, 329, 115, 58, 327, 267, 1722]))
    assert torch.equal(y, torch.tensor([623, 329, 115, 58, 327, 267, 1722, 2197]))


def test_dataset_and_train():
    config = configuration()
    config.block_size = 8
    config.training_iterations_per_epoch = 1
    config.validation_iterations_per_epoch = 1
    config.count_epochs = 1

    model = GPT(config=config)
    model.summarize_parameters()

    training_dataset, validation_dataset = InMemoryTokenDataset.load(
        filename=fixture_loader.path_to_file("corpus.txt"),
        vocab_filename=fixture_loader.path_to_file("vocab.json"),
        merge_filename=fixture_loader.path_to_file("merges.bpe"),
        block_size=config.block_size,
        device=torch.device("cpu"),
    )

    train(model=model, training_dataset=training_dataset, validation_dataset=validation_dataset)
