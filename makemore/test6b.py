import torch

list_of_lists = [[3.0, 2.0, 1.0], [4.0, 5.0, 6.0], [9.0, 7.0, 8.0]]
t1 = torch.tensor(list_of_lists)
_, indices = t1.max(dim=1, keepdim=True)
d = torch.zeros_like(t1).scatter(1, indices, 1.0)
print(d)

# t1.max(dim=1)
