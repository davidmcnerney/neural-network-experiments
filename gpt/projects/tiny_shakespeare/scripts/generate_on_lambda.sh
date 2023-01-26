# This script is to be run on the lambda instance itself.

set -e

VOCAB_FILENAME=~/project/resources/tiny_shakespeare.5000.vocab.json
MERGE_FILENAME=~/project/resources/tiny_shakespeare.5000.merge.bpe
MODEL_FILENAME=~/project/tiny_shakespeare.model
PROMPT="If only thou wouldst"

cd ~/project/code/
PYTHONPATH=.  python gpt/projects/tiny_shakespeare/generate_tool.py --vocab-file "$VOCAB_FILENAME" --merge-file "$MERGE_FILENAME" --model-file "$MODEL_FILENAME" --prompt "$PROMPT"
