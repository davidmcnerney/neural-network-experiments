import random
from typing import Dict, List, Optional, Set

import torch
from torch.nn import functional as F


# Video 2: more sophisticated neural net to predict next char of a name, based on BLOCK_SIZE preceding chars


# Hyperparameters etc
BLOCK_SIZE = 3  # how many preceding characters we use as X inputs to predict with
CHARACTER_DIMENSIONS = 2  # how many numbers we use to represent a character
LAYER_1_COUNT_NEURONS = 100
LEARNING_RATE = 0.1
BATCH_SIZE = 32
TRAINING_CYCLES = 10000

# Misc constants
EDGE_MARKER = "."  # depends on this character not appearing in the names.txt file

# Load names from the file
BASE_PATH = "/Users/dave/Dropbox/Projects/Learning/Neural Networks/Karpathy/neural-network-learning-karpathy"
names_file = f"{BASE_PATH}/resources/names.txt"
names = open(names_file).read().splitlines()
# names = names[:5]  # limit data set for dev work
random.seed(42)
random.shuffle(names)
print(f"{len(names)} names in input data set.")

# We'll represent characters by integer codes
char_set: Set[str] = set()
for name in names:
    for char in name:
        char_set.add(char)
char_set.add(EDGE_MARKER)
chars = sorted(char_set)
# Hard-code the char set for now, to perfectly match numbers in the videos.
# chars = [EDGE_MARKER, "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
# assert len(chars) == 27
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
count_training = round(len(xs) * 0.8)
count_dev = round(len(xs) * 0.1)
count_test = len(xs) - count_dev - count_training
n1 = count_training
n2 = count_training + count_dev
xs_training = xs[:n1]
ys_training = ys[:n1]
xs_dev = xs[n1:n2]
ys_dev = ys[n1:n2]
xs_test = xs[n2:]
ys_test = ys[n2:]
assert len(xs_training) + len(xs_dev) + len(xs_test) == len(xs)
assert len(ys_training) + len(ys_dev) + len(ys_test) == len(ys)
assert len(xs_training) == len(ys_training)
assert len(xs_dev) == len(ys_dev)
X_training = torch.tensor(xs_training)
Y_training = torch.tensor(ys_training)
X_dev = torch.tensor(xs_dev)
Y_dev = torch.tensor(ys_dev)
X_test = torch.tensor(xs_test)
Y_test = torch.tensor(ys_test)
print(f"Data sets - total: {len(xs)} training: {len(xs_training)} dev: {len(xs_dev)} test: {len(xs_test)}")

# Initial lookup matrix
# Maps character code one-hot vectors to vectors of CHARACTER_DIMENSIONS size
# Each row N is the vector for character N, so you can directly index as well
generator = torch.Generator().manual_seed(2147483647)
C = torch.randn((len(chars), CHARACTER_DIMENSIONS), generator=generator)

# Layer 1
# Maps the combined character vectors for the characters in the preceding block to vector containing one output
# float per neuron
W1 = torch.randn((CHARACTER_DIMENSIONS * BLOCK_SIZE, LAYER_1_COUNT_NEURONS), generator=generator)
b1 = torch.randn(LAYER_1_COUNT_NEURONS, generator=generator)

# Layer 2
# Maps layer 1 output to probability vector of size len(chars)
W2 = torch.randn((LAYER_1_COUNT_NEURONS, len(chars)), generator=generator)
b2 = torch.randn(len(chars), generator=generator)

# Require grad for our leaf parameters.
# Must be done before the forward pass, in order for the operations we perform to have a grad function
# attached to them
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
print(f"{sum(p.nelement() for p in parameters)} trainable parameters in the model.")


def forward_pass(X_: torch.Tensor, Y_: torch.Tensor) -> torch.Tensor:
    l1_out = torch.tanh(C[X_].view(-1, CHARACTER_DIMENSIONS * BLOCK_SIZE) @ W1 + b1)
    logits = l1_out @ W2 + b2
    loss_ = F.cross_entropy(logits, Y_)
    return loss_


def backward_pass(loss_: torch.Tensor) -> None:
    for p in parameters:
        p.grad = None
    loss_.backward()
    for p in parameters:
        p.data += -LEARNING_RATE * p.grad


for _ in range(TRAINING_CYCLES):
    # Obtain this batch
    batch_indices = torch.randint(0, X_training.shape[0], (BATCH_SIZE,), generator=generator)
    X_batch = X_training[batch_indices]
    Y_batch = Y_training[batch_indices]

    # Forward pass
    loss = forward_pass(X_batch, Y_batch)
    if TRAINING_CYCLES <= 100:
        print(f"Batch loss: {loss.item()}")

    # Backward pass
    backward_pass(loss)

# Forward pass with the different slices
print("")
training_loss = forward_pass(X_training, Y_training)
print(f"Training loss: {training_loss}")
dev_loss = forward_pass(X_dev, Y_dev)
print(f"Dev loss: {dev_loss}")
