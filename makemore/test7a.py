import torch

t1 = torch.tensor([
    [3.0, 2.0, 1.0],
    [4.0, 5.0, 6.0],
    [9.0, 7.0, 8.0],
])
print(t1.mean(dim=0, keepdim=True))
