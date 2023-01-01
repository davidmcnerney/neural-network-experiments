import torch
import torch.nn.functional as F


t1 = torch.tensor([
    [1., 1., 1., 1.],
    [2., 2., 2., 2.],
    [3., 3., 3., 3.],
])
print(t1)
print(t1[:, None])
# print(t1.T)

# print(F.softmax(t1, dim=-1))