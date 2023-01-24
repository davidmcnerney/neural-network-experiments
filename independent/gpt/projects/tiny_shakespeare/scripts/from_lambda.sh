set -e

IP_ADDRESS=$1
DESTINATION_MODEL_NAME=$2

SOURCE_MODEL_NAME="tiny_shakespeare.model"

# Copy model back
scp -r "ubuntu@$IP_ADDRESS:project/$SOURCE_MODEL_NAME" "$HOME/Temp/gpt/models/$DESTINATION_MODEL_NAME"
