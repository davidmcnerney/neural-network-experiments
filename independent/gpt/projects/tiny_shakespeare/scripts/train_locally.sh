set -e

CORPUS_FILENAME=~/Temp/gpt/bpe/tiny_shakespeare.txt
VOCAB_FILENAME=~/Temp/gpt/bpe/tiny_shakespeare.5000.vocab.json
MERGE_FILENAME=~/Temp/gpt/bpe/tiny_shakespeare.5000.merge.bpe
MODEL_FILENAME=~/Temp/gpt/models/tiny_shakespeare.model

PYTHONPATH=.  pipenv run python independent/gpt/projects/tiny_shakespeare/train_tool.py --corpus-file "$CORPUS_FILENAME" --vocab-file "$VOCAB_FILENAME" --merge-file "$MERGE_FILENAME" --model-file "$MODEL_FILENAME"
