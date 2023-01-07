import torch
from torch.nn import functional as F


# logits for:
#   batch size: 2
#   seq length: 3
#   vocab size: 4
t = torch.tensor([
    [                                   # batch sample #1
        [111, 112, 113, 114],              # 3 tokens in sequence, 4 dimensional embeddings
        [121, 122, 123, 124],
        [131, 132, 133, 134],
    ],
    [                                   # batch sample #2
        [211, 212, 213, 214],               # 3 tokens in sequence, 4 dimensional embeddings
        [221, 222, 223, 224],
        [231, 232, 233, 234],
    ],
])

# Pluck last logits
# t = t[:, -1:, :]

# View tests
t = t.view(-1, t.size(-1))

print(t)
print(t.shape)
