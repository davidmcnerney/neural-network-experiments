from typing import Dict, List, Set

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

# Neuron
# 5x27 input matrix multiplied by a 27x27 matrix yields 5x27 output matrix, where output rows correspond to input rows
W = torch.randn((27, 27))

# Forward pass
logits = (xenc @ W)         # each col represents probability of that next char; we consider it log of predicted counts
counts = logits.exp()       # after e**x operation, each col is proportional to predicted counts
sums = counts.sum(dim=1, keepdims=True)   # collapses columns (dim 1) so that we get a 1 column matrix, whose value for each row is the sum of the elements in that row
probs = counts / sums    # uses broadcasting to divide each column of each row by the one value in that row from the `sums` 5x1 matrix

print("Done")