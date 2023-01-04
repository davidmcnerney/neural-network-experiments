import torch


block_size = 4

t = torch.ones(block_size, block_size)
t = torch.tril(t)
# t = t.view(1, 1, block_size, block_size)
t = t.unsqueeze(0).unsqueeze(0)
print(t)

sequence_length = 3
t = t[:, :, :sequence_length, :sequence_length]
print(t)

x = torch.randn([1, 1, sequence_length , sequence_length])
print (x)

x = x.masked_fill(t == 0, float("-inf"))
print(x)