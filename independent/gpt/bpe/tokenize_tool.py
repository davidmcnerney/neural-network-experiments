import argparse

from independent.gpt.bpe import tokenizer
from independent.gpt.bpe import input_output


# Usage:
#   PYTHONPATH=.  pipenv run python independent/gpt/bpe/tokenize_tool.py --input-file ~/Temp/gpt/bpe/tinyshakespeare.txt --vocab-file ~/Temp/gpt/bpe/tinyshakespeare.5000.vocab.json --merge-file ~/Temp/gpt/bpe/tinyshakespeare.5000.merge.bpe --output-file ~/Temp/gpt/bpe/tinyshakespeare.encoded_5000.txt


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser("Build BPE")
    argument_parser.add_argument("--input-file", type=str, required=True)
    argument_parser.add_argument("--vocab-file", type=str, required=True)
    argument_parser.add_argument("--merge-file", type=str, required=True)
    argument_parser.add_argument("--output-file", type=str, required=True)
    args = argument_parser.parse_args()

    with open(args.input_file) as input_file:
        input_text = input_file.read()

    vocab = input_output.load_vocabulary_from_file(args.vocab_file)
    merges = input_output.load_merge_list_from_file(args.merge_file)

    output_tokens = tokenizer.tokenize(input_text, vocab, merges)
    with open(args.output_file, "w") as output_file:
        for token in output_tokens:
            output_file.write(str(token) + "\n")

    print(f"Loaded input: {len(input_text)} chars")
    print(f"Wrote output: {len(output_tokens)} tokens")
