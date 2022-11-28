import torch


t1 = torch.randn(32, 4, 68)
print(t1.shape)
t1a = t1.mean(dim=0, keepdim=True)
print(t1a.shape)
print(t1a)