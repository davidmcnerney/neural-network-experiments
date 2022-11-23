import torch

# list_of_lists = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

x = [13.0]
t1 = torch.tensor(x)
t1.requires_grad = True

t2 = t1 ** 2
t2.retain_grad()

out = (2.0 * t2).mean()
out.backward()

print(f"t1.grad: {t1.grad}")  # expect 2.0 * 2.0 * 13.0 = 52.0
print(f"t2.grad: {t2.grad}")  # expect 2.0
