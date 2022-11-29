import torch
from torch import nn

model = nn.Sequential(
    nn.Embedding(27, 10),
)
num_parameters = len(list(model.parameters()))
print(f"num_parameters: {num_parameters}")
