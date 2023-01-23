import argparse
from typing import Tuple

import torch

from independent.gpt.model.gpt import GPT
from independent.gpt.projects.tiny_shakespeare.configuration import configuration
from independent.gpt.running.dataset import InMemoryTokenDataset
from independent.gpt.running.train import train


# Usage:
#   PYTHONPATH=.  pipenv run python independent/gpt/projects/tiny_shakespeare/train_tool.py --model-file /Users/dave/Temp/gpt/models/tinyshakespeare.model


def parse_arguments() -> Tuple[str, str]:
    argument_parser = argparse.ArgumentParser("Train")
    argument_parser.add_argument("--model-file", type=str, required=True)
    argument_parser.add_argument("--continue", action=argparse.BooleanOptionalAction, dest="continue_training")
    args = argument_parser.parse_args()
    return args.model_file, args.continue_training


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    model_filename, continue_training = parse_arguments()

    if continue_training:
        model = torch.load(model_filename)
    else:
        config = configuration()
        model = GPT(config=config)
    model.summarize_parameters()

    device = get_device()
    model.to(device)
    print(f"Using device: {device}")
    # TODO: may need to use map_location in torch.load above to avoid errors if continuing training on different hardware?

    training_dataset, validation_dataset = InMemoryTokenDataset.load(
        filename="/Users/dave/Temp/gpt/bpe/tinyshakespeare.txt",
        vocab_filename="/Users/dave/Temp/gpt/bpe/tinyshakespeare.5000.vocab.json",
        merge_filename="/Users/dave/Temp/gpt/bpe/tinyshakespeare.5000.merge.bpe",
        block_size=model.config.block_size,
        device=device,
    )

    train(
        model=model,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model_save_filename=model_filename,
    )
