import argparse
from typing import Tuple

import torch

from independent.gpt.model.gpt import GPT
from independent.gpt.projects.tiny_shakespeare.configuration import configuration
from independent.gpt.running.dataset import InMemoryTokenDataset
from independent.gpt.running.train import train


# Usage:
#   PYTHONPATH=.  pipenv run python independent/gpt/projects/tiny_shakespeare/train_tool.py --corpus-file /Users/dave/Temp/gpt/bpe/tiny_shakespeare.txt --vocab-file /Users/dave/Temp/gpt/bpe/tiny_shakespeare.5000.vocab.json --merge-file /Users/dave/Temp/gpt/bpe/tiny_shakespeare.5000.merge.bpe --model-file /Users/dave/Temp/gpt/models/tiny_shakespeare.model


def parse_arguments() -> Tuple[str, str, str, str, str]:
    argument_parser = argparse.ArgumentParser("Train")
    argument_parser.add_argument("--corpus-file", type=str, required=True)
    argument_parser.add_argument("--vocab-file", type=str, required=True)
    argument_parser.add_argument("--merge-file", type=str, required=True)
    argument_parser.add_argument("--model-file", type=str, required=True)
    argument_parser.add_argument("--continue", action="store_true", dest="continue_training")
    argument_parser.set_defaults(continue_training=False)
    args = argument_parser.parse_args()
    return args.corpus_file, args.vocab_file, args.merge_file, args.model_file, args.continue_training


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    corpus_filename, vocab_filename, merge_filename, model_filename, continue_training = parse_arguments()

    if continue_training:
        print("Continuing previous training.")
        model = torch.load(model_filename)
    else:
        print("Training from scratch.")
        config = configuration()
        model = GPT(config=config)
    model.summarize_parameters()

    device = get_device()
    model.to(device)
    print(f"Using device: {device}")
    # TODO: may need to use map_location in torch.load above to avoid errors if continuing training on different hardware?

    training_dataset, validation_dataset = InMemoryTokenDataset.load(
        filename=corpus_filename,
        vocab_filename=vocab_filename,
        merge_filename=merge_filename,
        block_size=model.config.block_size,
        device=device,
    )

    train(
        model=model,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        model_save_filename=model_filename,
    )
