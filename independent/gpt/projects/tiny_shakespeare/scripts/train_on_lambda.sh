# This script is to be run on the lambda instance itself.

set -e

CORPUS_FILENAME=~/project/resources/tiny_shakespeare.txt
VOCAB_FILENAME=~/project/resources/tiny_shakespeare.5000.vocab.json
MERGE_FILENAME=~/project/resources/tiny_shakespeare.5000.merge.bpe
MODEL_FILENAME=~/project/tiny_shakespeare.model

cd ~/project/code/
PYTHONPATH=.  python independent/gpt/projects/tiny_shakespeare/train_tool.py --corpus-file "$CORPUS_FILENAME" --vocab-file "$VOCAB_FILENAME" --merge-file "$MERGE_FILENAME" --model-file "$MODEL_FILENAME"
