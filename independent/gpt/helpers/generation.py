import torch
from torch.nn import functional as F


def top_p(
        logits: torch.Tensor,
        top_p: float,
) -> None:
    """
    Sets less probable elements of logits to -inf, such that the remaining elements have softmax probabilities
    which sum to less than the value for top_p provided. Always retains at least the highest-probability element.
    Operates across the last dimension. Commonly, a 2 or 3 dimensional logits tensor will be passed, e.g.:
    batch size x sequence length x vocab size.
    """
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    softmax_sorted_logits = F.softmax(sorted_logits, dim=-1)
    cumulative_softmax_sorted_logits = torch.cumsum(softmax_sorted_logits, dim=-1)
    sorted_inclusion_status = cumulative_softmax_sorted_logits < top_p
    sorted_inclusion_status[..., 0] = True   # always retain highest-probability element
    inclusion_status = torch.gather(sorted_inclusion_status, -1, sorted_indices.argsort(-1))
    logits[~inclusion_status] = -float("inf")
