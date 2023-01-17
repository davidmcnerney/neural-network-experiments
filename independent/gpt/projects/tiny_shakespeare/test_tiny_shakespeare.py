from independent.gpt.projects.tiny_shakespeare.configuration import config
from independent.gpt.projects.tiny_shakespeare.dataset import TinyShakespeareDataset
from independent.gpt.projects.tiny_shakespeare.test_fixtures.fixture_loader import fixture_loader


def test_dataset():
    dataset = TinyShakespeareDataset(
        filename=fixture_loader.path_to_file("corpus.txt"),
        vocab_filename=fixture_loader.path_to_file("vocab.json"),
        merge_filename=fixture_loader.path_to_file("merges.txt"),
        block_size=config.block_size,
    )
