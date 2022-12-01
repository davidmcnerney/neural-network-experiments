import torch
from torch import nn

layer = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=5)
print(list([p for p in layer.parameters() if p.requires_grad]))
nelements = sum(p.numel() for p in layer.parameters() if p.requires_grad)
print(f"has {nelements} elements")

# t1 = torch.tensor([
#     [3.0, 2.0, 1.0],
#     [4.0, 5.0, 6.0],
#     [9.0, 7.0, 8.0],
# ])
# m = nn.Flatten(1)
# print(m(t1))
# print(torch.flatten(t1))

# t2 = torch.randn(15, 16, 4, 4)
# # print(t2)
# # t2f = t2.flatten(start_dim=1)
# t2f = nn.Flatten()(t2)
# print(t2f.shape)
# print(t2f)
