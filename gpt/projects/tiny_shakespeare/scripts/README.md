Procedure for training on Lambda Labs instance:
- Create instance at https://cloud.lambdalabs.com/instances, note the IP
- `> cd /Users/dave/Projects/neural-network-experiments/`
- `> bash gpt/projects/tiny_shakespeare/scripts/to_lambda.sh <IP>`
- `> ssh ubuntu@<IP>`
- `ubuntu@<IP>> bash project/code/gpt/projects/tiny_shakespeare/scripts/train_on_lambda.sh`
- `ubuntu@<IP>> bash project/code/gpt/projects/tiny_shakespeare/scripts/generate_on_lambda.sh`
- `ubuntu@<IP>> bash project/code/gpt/projects/tiny_shakespeare/scripts/train_on_lambda.sh --continue --count-epochs 500`

Some of the scripts have hardcoded paths on my local machine for BPE encoding and model files, so will need to be updated for other users.
