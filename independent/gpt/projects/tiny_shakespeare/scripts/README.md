Procedure for training on Lambda Labs instance:
- Create instance at https://cloud.lambdalabs.com/instances, note the IP
- `> cd "/Users/dave/Dropbox/Projects/Learning/Neural Networks/Karpathy/neural-network-learning-karpathy/"`
- `> bash independent/gpt/projects/tiny_shakespeare/scripts/to_lambda.sh <IP>`
- `> bash independent/gpt/projects/tiny_shakespeare/scripts/kick_off_training_on_lambda.sh <IP>`
- In the remote shell, rerun training as needed with `PYTHONPATH=.  python independent/gpt/projects/tiny_shakespeare/train_tool.py --model-file ~/project/tiny_shakespeare.model --continue`
- `> bash independent/gpt/projects/tiny_shakespeare/scripts/from_lambda.sh <IP> ~/Temp/gpt/models/tiny_shakespeare.X.model`
