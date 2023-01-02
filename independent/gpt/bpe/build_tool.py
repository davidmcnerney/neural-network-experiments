import argparse

from independent.gpt.bpe import builder


# Usage:
#   PYTHONPATH=.  pipenv run python independent/gpt/bpe/build_tool.py --input-file ~/Temp/OpenAI/bpe/tinyshakespeare.txt --count-merges 1000


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser("Build BPE")
    argument_parser.add_argument("--input-file", type=str, required=True)
    argument_parser.add_argument("--count-merges", type=int, required=True)
    args = argument_parser.parse_args()

    with open(args.input_file) as fh:
        text = fh.read()
    vocab, merges = builder.build_vocabulary_and_merge_list(training_text=text, count_merges=args.count_merges)

    print("\nVocabulary:")
    builder.summarize_vocab(vocab)
    print("\nMerges:")
    builder.summarize_merges(merges)
