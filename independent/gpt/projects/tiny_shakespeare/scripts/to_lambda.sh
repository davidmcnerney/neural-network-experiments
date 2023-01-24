set -e

IP_ADDRESS=$1

# Create folders
ssh "ubuntu@$IP_ADDRESS" "mkdir -p project/code/independent/"
ssh "ubuntu@$IP_ADDRESS" "mkdir -p project/resources/"

# Copy code, BPE files, and text corpus
scp -r independent/gpt "ubuntu@$IP_ADDRESS:project/code/independent"   # copies pycache, pytest cache etc
scp ~/Temp/gpt/bpe/tiny_shakespeare.txt "ubuntu@$IP_ADDRESS:project/resources"
scp ~/Temp/gpt/bpe/tiny_shakespeare.5000.vocab.json "ubuntu@$IP_ADDRESS:project/resources"
scp ~/Temp/gpt/bpe/tiny_shakespeare.5000.merge.bpe "ubuntu@$IP_ADDRESS:project/resources"

# Set up environment
ssh "ubuntu@$IP_ADDRESS" "/usr/bin/pip install regex"
