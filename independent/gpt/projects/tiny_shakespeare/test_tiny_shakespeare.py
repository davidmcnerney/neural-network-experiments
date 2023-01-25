import torch

from independent.gpt.model.gpt import GPT
from independent.gpt.projects.tiny_shakespeare.configuration import configuration
from independent.gpt.projects.tiny_shakespeare.test_fixtures.fixture_loader import fixture_loader
from independent.gpt.running.dataset import InMemoryTokenDataset
from independent.gpt.running.train import train


# encoded.txt was generated with this command:
# PYTHONPATH=.  pipenv run python independent/gpt/bpe/tokenize_tool.py --input-file independent/gpt/projects/tiny_shakespeare/test_fixtures/corpus.txt --vocab-file independent/gpt/projects/tiny_shakespeare/test_fixtures/vocab.json --merge-file independent/gpt/projects/tiny_shakespeare/test_fixtures/merges.bpe --output-file independent/gpt/projects/tiny_shakespeare/test_fixtures/encoded.txt


def test_dataset():
    training_dataset, validation_dataset = InMemoryTokenDataset.load(
        filename=fixture_loader.path_to_file("corpus.txt"),
        vocab_filename=fixture_loader.path_to_file("vocab.json"),
        merge_filename=fixture_loader.path_to_file("merges.bpe"),
        block_size=8,
        device=torch.device("cpu"),
        mix_training_and_validation=False,  # so that we know which blocks will be first for validation
    )

    # 311 tokens -> 38 blocks -> 3 validation blocks, 35 training blocks

    # Training dataset first block begins at first token in encoded.txt
    assert len(training_dataset) == 34
    x, y = training_dataset[0]
    assert torch.equal(x, torch.tensor([684, 1244, 58, 10, 2562, 342, 3023, 2598]))  # these are from encoded.txt
    assert torch.equal(y, torch.tensor([1244, 58, 10, 2562, 342, 3023, 2598, 2524]))

    # Validation dataset first block begins at token 8 * 34 + 1 = 273rd token
    assert len(validation_dataset) == 4
    x, y = validation_dataset[0]
    assert torch.equal(x, torch.tensor([10, 382, 289, 1046, 115, 44, 1371, 342]))
    assert torch.equal(y, torch.tensor([382, 289, 1046, 115, 44, 1371, 342, 2440]))


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
