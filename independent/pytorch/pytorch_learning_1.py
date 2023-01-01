import torch

batch_size = 3
seq_size = 4
vocab_size = 5

l = torch.rand([batch_size, seq_size, vocab_size])
print(l)
l2 = l[:,-1,:]
print(l2)