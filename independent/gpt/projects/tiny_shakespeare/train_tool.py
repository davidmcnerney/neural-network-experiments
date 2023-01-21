import argparse

import torch

from independent.gpt.model.gpt import GPT
from independent.gpt.projects.tiny_shakespeare.configuration import configuration
from independent.gpt.running.dataset import InMemoryTokenDataset
from independent.gpt.running.training import train


# Usage:
#   PYTHONPATH=.  pipenv run python independent/gpt/projects/tiny_shakespeare/train_tool.py --model-file ~/Temp/gpt/models/tinyshakespeare.model


def _parse_arguments() -> str:
    """
    Returns model filename
    """
    argument_parser = argparse.ArgumentParser("Train")
    argument_parser.add_argument("--model-file", type=str, required=True)
    return argument_parser.parse_args().model_file


if __name__ == "__main__":
    # TODO: call .to() to move model to device, to support use of GPU

    model_filename = _parse_arguments()

    config = configuration()
    model = GPT(config=config)
    # TODO: we want to have training and test datasets, both passed to train, and report test dataset loss on each iteration
    training_dataset, validation_dataset = InMemoryTokenDataset.load(
        filename="/Users/dave/Temp/gpt/bpe/tinyshakespeare.encoded_5000.txt",
        vocab_filename="/Users/dave/Temp/gpt/bpe/tinyshakespeare.5000.vocab.json",
        merge_filename="/Users/dave/Temp/gpt/bpe/tinyshakespeare.5000.merge.bpe",
        block_size=config.block_size,
    )

    train(
        model=model,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
    )

    torch.save(model, model_filename)
    print("Saved model.")
