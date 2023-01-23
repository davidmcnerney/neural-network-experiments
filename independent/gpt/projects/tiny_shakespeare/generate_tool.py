import argparse
from typing import Tuple

import torch

from independent.gpt.bpe.input_output import load_merge_list_from_file, load_vocabulary_from_file
from independent.gpt.running.generate import generate


# Usage:
#   PYTHONPATH=.  pipenv run python independent/gpt/projects/tiny_shakespeare/generate_tool.py --model-file /Users/dave/Temp/gpt/models/tinyshakespeare.model --prompt "If only thou wouldst"


def parse_arguments() -> Tuple[str, str]:
    argument_parser = argparse.ArgumentParser("Generate")
    argument_parser.add_argument("--model-file", type=str, required=True)
    argument_parser.add_argument("--prompt", type=str, required=True)
    args = argument_parser.parse_args()
    return args.model_file, args.prompt


if __name__ == "__main__":
    model_filename, prompt = parse_arguments()
    completion = generate(
        model=torch.load(model_filename),
        vocabulary_by_token=load_vocabulary_from_file("/Users/dave/Temp/gpt/bpe/tinyshakespeare.5000.vocab.json"),
        merge_list=load_merge_list_from_file("/Users/dave/Temp/gpt/bpe/tinyshakespeare.5000.merge.bpe"),
        prompt=prompt,
        max_output_tokens=1000,
    )
    print(completion)
