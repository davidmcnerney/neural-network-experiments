import torch

from independent.gpt.model.gpt import GPT
from independent.gpt.projects.tiny_shakespeare.configuration import configuration
from independent.gpt.running.dataset import InMemoryTokenDataset
from independent.gpt.running.train import train


# Usage:
#   PYTHONPATH=.  pipenv run python independent/gpt/projects/tiny_shakespeare/train_tool.py


if __name__ == "__main__":
    # TODO: call .to() to move model to device, to support use of GPU

    config = configuration()
    model = GPT(config=config)
    training_dataset, validation_dataset = InMemoryTokenDataset.load(
        filename="/Users/dave/Temp/gpt/bpe/tinyshakespeare.txt",
        vocab_filename="/Users/dave/Temp/gpt/bpe/tinyshakespeare.5000.vocab.json",
        merge_filename="/Users/dave/Temp/gpt/bpe/tinyshakespeare.5000.merge.bpe",
        block_size=config.block_size,
    )

    train(
        model=model,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
    )

    torch.save(model, "/Users/dave/Temp/gpt/models/tinyshakespeare.model")
    print("Saved model.")
