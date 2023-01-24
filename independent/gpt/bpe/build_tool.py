import argparse

from independent.gpt.bpe import builder
from independent.gpt.bpe import input_output


# Usage:
#   PYTHONPATH=.  pipenv run python independent/gpt/bpe/build_tool.py --input-file ~/Temp/gpt/bpe/tiny_shakespeare.txt --count-merges 5000 --output-vocab-file ~/Temp/gpt/bpe/tiny_shakespeare.5000.vocab.json --output-merge-file ~/Temp/gpt/bpe/tiny_shakespeare.5000.merge.bpe


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser("Build BPE")
    argument_parser.add_argument("--input-file", type=str, required=True)
    argument_parser.add_argument("--count-merges", type=int, required=True)
    argument_parser.add_argument("--output-vocab-file", type=str, required=True)
    argument_parser.add_argument("--output-merge-file", type=str, required=True)
    args = argument_parser.parse_args()

    with open(args.input_file) as file:
        text = file.read()
    vocab, merges = builder.build_vocabulary_and_merge_list(training_text=text, count_merges=args.count_merges)

    # print("\nVocabulary:")
    # builder.summarize_vocab(vocab)
    # print("\nMerges:")
    # builder.summarize_merges(merges)

    with open(args.output_vocab_file, "w") as file:
        file.write(input_output.serialize_vocabulary(vocab))
    with open(args.output_merge_file, "w") as file:
        file.write(input_output.serialize_merge_list(merges))
    print("\nWrote files.")
