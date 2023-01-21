from independent.gpt.model.gpt import GPT
from independent.gpt.projects.tiny_shakespeare.configuration import configuration
from independent.gpt.projects.tiny_shakespeare.dataset import TinyShakespeareDataset
from independent.gpt.running.training import train


# Usage:
#   PYTHONPATH=.  pipenv run python independent/gpt/projects/tiny_shakespeare/train_tool.py


if __name__ == "__main__":
    config = configuration()
    model = GPT(config=config)
    # TODO: call .to() to move model to device, to support use of GPU
    dataset = TinyShakespeareDataset(
        filename="/Users/dave/Temp/gpt/bpe/tinyshakespeare.encoded_5000.txt",
        vocab_filename="/Users/dave/Temp/gpt/bpe/tinyshakespeare.5000.vocab.json",
        merge_filename="/Users/dave/Temp/gpt/bpe/tinyshakespeare.5000.merge.bpe",
        block_size=config.block_size,
    )
    train(
        model=model,
        dataset=dataset,
    )
    # TODO: save the model for use in inference
    #    - maybe save checkpoints
