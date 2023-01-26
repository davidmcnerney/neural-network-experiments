set -e

IP_ADDRESS=$1

ssh -t "ubuntu@$IP_ADDRESS" "bash project/code/gpt/projects/tiny_shakespeare/scripts/train_on_lambda.sh; exec bash -l"
