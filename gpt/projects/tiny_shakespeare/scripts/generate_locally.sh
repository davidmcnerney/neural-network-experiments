set -e

VOCAB_FILENAME=~/Temp/gpt/bpe/tiny_shakespeare.5000.vocab.json
MERGE_FILENAME=~/Temp/gpt/bpe/tiny_shakespeare.5000.merge.bpe
MODEL_FILENAME=~/Temp/gpt/models/tiny_shakespeare.model
PROMPT="If only thou wouldst"

PYTHONPATH=.  pipenv run python gpt/projects/tiny_shakespeare/generate_tool.py --vocab-file "$VOCAB_FILENAME" --merge-file "$MERGE_FILENAME" --model-file "$MODEL_FILENAME" --prompt "$PROMPT"
