import argparse
import json
from typing import Dict

from independent.gpt.bpe import helpers
from independent.gpt.bpe import input_output
from independent.gpt.bpe import type_definitions


# Usage:
#   PYTHONPATH=.  pipenv run python independent/gpt/bpe/invert_vocab_tool.py --input-vocab-file ~/Temp/gpt/bpe/tiny_shakespeare.10000.vocab.inverted.json --output-vocab-file ~/Temp/gpt/bpe/tiny_shakespeare.10000.vocab.json


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser("Invert vocab")
    argument_parser.add_argument("--input-vocab-file", type=str, required=True)
    argument_parser.add_argument("--output-vocab-file", type=str, required=True)
    args = argument_parser.parse_args()

    with open(args.input_vocab_file) as file:
        in_text = file.read()

    inverted_vocab: Dict[str, str] = json.loads(in_text)
    vocab_with_string_indices: Dict[str, str] = helpers.invert_dictionary(inverted_vocab)
    vocab: type_definitions.VocabularyByToken = {k: int(v) for k, v in vocab_with_string_indices.items()}

    with open(args.output_vocab_file, "w") as file:
        file.write(input_output.serialize_vocabulary(vocab))

    print("\nWrote file.")
