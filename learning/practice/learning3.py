import torch
from torch.nn import functional as F


t = torch.tensor([
    [1, 1, 500],
    [1, 2, 1000],
    [3, 3, 3],
    [1, 4, 0],
], dtype=torch.float)
t = F.softmax(t, dim=-1)
print(t)
