import torch

from independent.gpt.model.gpt import GPT
from independent.gpt.running.configuration import Configuration

def test_forward():
    x = torch.tensor([[100, 200, 300, 400]])   # token indices: batch size x seq length
    gpt = GPT(config=Configuration())
    y = gpt(x)
    print(y.shape)

