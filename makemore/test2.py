import torch

t1 = torch.tensor([
    [1.0, 2.0, 3.0],
])
t2 = torch.tensor([
    [1.0, 2.0],
    [1.0, 2.0],
    [1.0, 2.0],
])
t3 = t1 @ t2


summed = t2.sum(dim=1, keepdim=True)
print("Done")
