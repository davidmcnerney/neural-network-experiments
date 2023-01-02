import argparse

from independent.gpt.bpe import builder


# Usage:
#   PYTHONPATH=.  pipenv run python independent/gpt/bpe/build_tool.py --input-file ~/Temp/OpenAI/bpe/tinyshakespeare.txt --count-merges 10000 --output-vocab-file ~/Temp/gpt/bpe/tinyshakespeare.10000.vocab.json --output-merge-file ~/Temp/gpt/bpe/tinyshakespeare.10000.merge.bpe


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
        file.write(builder.serialize_vocab(vocab))
    with open(args.output_merge_file, "w") as file:
        file.write(builder.serialize_merges(merges))
    print("\nWrote files.")
