import argparse
from typing import Tuple

import torch

from gpt.bpe.input_output import load_merge_list_from_file, load_vocabulary_from_file
from gpt.running.generate import generate


# Usage:
#   PYTHONPATH=.  pipenv run python gpt/projects/tiny_shakespeare/generate_tool.py --vocab-file /Users/dave/Temp/gpt/bpe/tiny_shakespeare.5000.vocab.json --merge-file /Users/dave/Temp/gpt/bpe/tiny_shakespeare.5000.merge.bpe --model-file /Users/dave/Temp/gpt/models/tiny_shakespeare.model --prompt "If only thou wouldst"


def parse_arguments() -> Tuple[str, str, str, str]:
    argument_parser = argparse.ArgumentParser("Generate")
    argument_parser.add_argument("--vocab-file", type=str, required=True)
    argument_parser.add_argument("--merge-file", type=str, required=True)
    argument_parser.add_argument("--model-file", type=str, required=True)
    argument_parser.add_argument("--prompt", type=str, required=True)
    args = argument_parser.parse_args()
    return args.vocab_file, args.merge_file, args.model_file, args.prompt


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    vocab_filename, merge_filename, model_filename, prompt = parse_arguments()

    device = get_device()

    completion = generate(
        model=torch.load(model_filename, map_location=device),
        vocabulary_by_token=load_vocabulary_from_file(vocab_filename),
        merge_list=load_merge_list_from_file(merge_filename),
        prompt=prompt,
        max_output_tokens=1000,
        device=device,
        top_p=0.95,
    )
    print(completion)
