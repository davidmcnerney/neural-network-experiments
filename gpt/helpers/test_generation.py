import torch

from gpt.helpers import generation


def test_top_p():
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

    generation.top_p(logits, 0.8)

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


def test_min_p():
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

    generation.min_p(logits, 0.1)

    expected_logits = torch.tensor([
        [
            [-float("inf"), -float("inf"), 3., 4.],
            [1., 1., 1., 1.],
            [1000., -float("inf"), -float("inf"), -float("inf")],
        ],
        [
            [-float("inf"), -float("inf"), -float("inf"), 10.],
            [-1., -1., -1., -1.],
            [-float("inf"), -float("inf"), -float("inf"), 10000.],
        ],
    ])
    assert torch.equal(logits, expected_logits)
