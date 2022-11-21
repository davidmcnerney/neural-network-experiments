from typing import Dict, List, Optional, Set

import torch
from torch.nn import functional as F


# Video 2: more sophisticated neural net to predict next char of a name, based on BLOCK_SIZE preceding chars


# Hyperparameters etc
BLOCK_SIZE = 3  # how many preceding characters we use as X inputs to predict with
CHARACTER_DIMENSIONS = 2  # how many numbers we use to represent a character
LAYER_1_COUNT_NEURONS = 100

# Misc constants
EDGE_MARKER = "."  # depends on this character not appearing in the names.txt file

# Load names from the file
BASE_PATH = "/Users/dave/Dropbox/Projects/Learning/Neural Networks/Karpathy/neural-network-learning-karpathy"
names_file = f"{BASE_PATH}/resources/names.txt"
names = open(names_file).read().splitlines()
names = names[:5]  # limit data set for dev work
print(f"{len(names)} names")

# We'll represent characters by integer codes
# char_set: Set[str] = set()
# for name in names:
#     for char in name:
#         char_set.add(char)
# char_set.add(EDGE_MARKER)
# chars = sorted(char_set)
chars = [EDGE_MARKER, "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
assert len(chars) == 27
char_to_code: Dict[str, int] = {char: code for code, char in enumerate(chars)}
code_to_char: Dict[int, str] = {code: char for code, char in enumerate(chars)}

# Create data set
# X inputs are BLOCK_SIZE element vectors, holding codes of the BLOCK_SIZE preceding chars.
# Y output is 1 element vector, HOLDING code of predicted next char
xs: List[List[int]] = []
ys: List[int] = []
for name in names:
    # print(name)
    # EDGE_MARKER represents positions before or after the name
    name = EDGE_MARKER * BLOCK_SIZE + name + EDGE_MARKER
    for pos in range(len(name) - BLOCK_SIZE):
        x_string = name[pos:pos+BLOCK_SIZE]   # BLOCK_SIZE characters preceding y_string
        y_string = name[pos+BLOCK_SIZE]       # 1 character
        x = [char_to_code[char] for char in x_string]
        y = char_to_code[y_string]
        xs.append(x)
        ys.append(y)
        # print(f"{''.join([code_to_char[code] for code in x])} ---> {code_to_char[y]}")
assert len(xs) == len(ys)
X = torch.tensor(xs)
Y = torch.tensor(ys)

# Divide into training, dev, and test data sets
# We divide our names data set up into 3 subsets: 80% for training, 10% for dev testing, and 10% for final test
# count_training = round(len(xs) * 0.8)
# count_dev = round(len(xs) * 0.1)
# count_test = len(xs) - count_dev - count_training
# xs_training = xs[:count_training]
# ys_training = ys[:count_training]
# xs_dev = xs[len(xs_training):len(xs_training) + count_dev]
# ys_dev = ys[len(ys_training):len(ys_training) + count_dev]
# xs_test = xs[len(xs_training) + len(xs_dev):]
# ys_test = ys[len(ys_training) + len(ys_dev):]
# assert len(xs_training) + len(xs_dev) + len(xs_test) == len(xs)
# assert len(ys_training) + len(ys_dev) + len(ys_test) == len(ys)
# assert len(xs_training) == len(ys_training)
# assert len(xs_dev) == len(ys_dev)
# print(f"total: {len(xs)} training: {len(xs_training)} dev: {len(xs_dev)} test: {len(xs_test)}")
# X_training = torch.tensor(xs_training)
# Y_training = torch.tensor(ys_training)
# X_dev = torch.tensor(xs_dev)
# Y_dev = torch.tensor(ys_dev)
# X_test = torch.tensor(xs_test)
# Y_test = torch.tensor(ys_test)

# Initial lookup matrix
# Maps character code one-hot vectors to vectors of CHARACTER_DIMENSIONS size
# Each row N is the vector for character N, so you can directly index as well
generator = torch.Generator().manual_seed(2147483647)
C = torch.randn((len(chars), CHARACTER_DIMENSIONS), generator=generator)

# Layer 1
# Maps the combined character vectors for the characters in the preceding block to vector containing one output
# float per neuron
W1 = torch.randn((CHARACTER_DIMENSIONS * BLOCK_SIZE, LAYER_1_COUNT_NEURONS), generator=generator)
print(f"W1: {W1.shape}")
b1 = torch.randn(LAYER_1_COUNT_NEURONS, generator=generator)
print(f"b1: {b1.shape}")

# Layer 2
# Maps layer 1 output to probability vector of size len(chars)
W2 = torch.randn((LAYER_1_COUNT_NEURONS, len(chars)), generator=generator)
b2 = torch.randn(len(chars), generator=generator)

parameters = [C, W1, b1, W2, b2]

# Forward pass
C_out = C[X]
print(f"C_out: {C_out.shape}")
# C_out is a 3 dimensional tensor of size len(X) x BLOCK_SIZE x CHARACTER_DIMENSIONS, for example
# 30 x 3 x 2. We need a 2 dimensional tensor of size len(X) x BLOCK_SIZE * CHARACTER_DIMENSIONS,
# in this example 30 x 6. We want to feed single vectors with all the character vectors just concatenated end-to-end
# into the first layer of the neural network. The view method does this efficiently for us.
C_out_flattened = C_out.view(-1, CHARACTER_DIMENSIONS * BLOCK_SIZE)
print(f"C_out_flattened: {C_out_flattened.shape}")
# b1 is broadcast to add its weights to every row of the output
l1_out = torch.tanh(C_out_flattened @ W1 + b1)
print(f"l1_out: {l1_out.shape}")
l2_out = l1_out @ W2 + b2
print(f"l2_out: {l2_out.shape}")
logits = l2_out
# Below is Softmax
counts = logits.exp()
sums = counts.sum(dim=1, keepdims=True)
print(f"sums: {sums.shape}")
probs = counts / sums
print(f"probs: {probs.shape}")
ar = torch.arange(Y.size(dim=0))
relevant = probs[ar, Y]
loss = -relevant.log().mean()
loss_amount = loss.item()

print("Done")
