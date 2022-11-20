from typing import Dict, List, Optional, Set

import torch
from torch.nn import functional as F


EDGE_MARKER = "."  # depends on this character not appearing in the names

# Load names from the file
BASE_PATH = "/Users/dave/Dropbox/Projects/Learning/Neural Networks/Karpathy/neural-network-learning-karpathy"
names_file = f"{BASE_PATH}/resources/names.txt"
names = open(names_file).read().splitlines()

# We'll represent characters by integer codes
char_set: Set[str] = set()
for name in names:
    for char in name:
        char_set.add(char)
char_set.add(EDGE_MARKER)
chars = sorted(char_set)
char_to_code: Dict[str, int] = {char: code for code, char in enumerate(chars)}
code_to_char: Dict[str, int] = {code: char for code, char in enumerate(chars)}

# Compute a training set from the data
x_list: List[int] = []   # latest character
y_list: List[int] = []   # next character after latest
for name in names[:1]:
    name = "." + name + "."
    for ch1, ch2 in zip(name, name[1:]):
        x_list.append(char_to_code[ch1])
        y_list.append(char_to_code[ch2])
xs = torch.tensor(x_list)
xenc = F.one_hot(xs, num_classes=27).float()
ys = torch.tensor(y_list)
yenc = F.one_hot(ys, num_classes=27).float()

# Create our one layer, with 27 neurons, one for each char
generator = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=generator, requires_grad=True)

# Training
loss_amount: Optional[float] = None
for i in range(10000):
    # Forward pass

    # Nx27 input matrix multiplied by a 27x27 matrix yields Nx27 output matrix, where output rows correspond to input rows
    # exp and normalization so as to sum to 1, as we do below, is called "softmax"
    logits = xenc @ W         # we interpret each col to be the log of predicted counts for corresponding char 0-27
    counts = logits.exp()       # after e**x operation, each col is now proportional to predicted counts
    sums = counts.sum(dim=1, keepdims=True)   # gives a Nx1 single column matrix, whose values are the sum of each row
    # (sum collapses columns (dimension 1) across each row to yield a row sum)
    probs = counts / sums       # now values in each row sum to exactly 1, and so are true relative probabilities
    # when we scalar divide a Nx1 matrix into a Nx27 matrix, broadcasting divides each value in a row by the 1 value
    # print(probs[0])

    # Calculate loss - we measure loss by the average negative log likelihood. For each x,y sample, we look up
    # the probability predicted by the network for the character y. If the network is predicting perfectly,
    # it would be 1.0. We take the log, which in perfect conditions will therefore be 0.0. Since we'll never be perfect,
    # we'll see probabilities < 1.0 which means negative logs, so we negate to get a positive number to minimize to 0.
    loss = -probs[torch.arange(len(ys)), ys].log().mean()
    loss_amount = loss.item()
    # print(f"{i+1:3} loss: {loss.item()}")

    # Backward pass
    W.grad = None
    loss.backward()
    W.data += -0.1 * W.grad

print(f"Final loss: {loss_amount}")

