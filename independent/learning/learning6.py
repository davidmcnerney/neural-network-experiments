import torch
from torch.nn import functional as F


# batch size x sequence length x vocab size
logits = torch.tensor([
    [  # batch example 1
        [1., 2., 3., 4.],       # token 1 in sequence
        [1., 1., 1., 1.],       # token 2
        [1000., 0., 0., 0.],    # token 3
    ],
    [  # batch example 2
        [1., 0., 0., 10.],       # token 1
        [-1., -1., -1., -1.],    # token 2
        [1., 2., 3., 10000.],    # token 3
    ],
])
top_p = 0.8
filter_value = -float("inf")

print(logits.shape)

sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
softmax_sorted_logits = F.softmax(sorted_logits, dim=-1)
cumulative_softmax_sorted_logits = torch.cumsum(softmax_sorted_logits, dim=-1)
sorted_inclusion_status = cumulative_softmax_sorted_logits < top_p
sorted_inclusion_status[..., 0] = True
inclusion_status = torch.gather(sorted_inclusion_status, -1, sorted_indices.argsort(-1))
logits[~inclusion_status] = filter_value

expected_logits = torch.tensor([
    [
        [-float("inf"), -float("inf"), -float("inf"), 4.],
        [1., 1., 1., -float("inf")],
        [1000., -float("inf"), -float("inf"), -float("inf")],
    ],
    [
        [-float("inf"), -float("inf"), -float("inf"), 10.],
        [-1., -1., -1., -float("inf")],
        [-float("inf"), -float("inf"), -float("inf"), 10000.],
    ],
])
assert torch.equal(logits, expected_logits)

print("Done")
