import random
import sys
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F


# Video 2: more sophisticated neural net to predict next char of a name, based on BLOCK_SIZE preceding chars
# to do:
#   - increase LAYER_1_COUNT_NEURONS
#   - improve C dimensions
#   - tune learning rate


# Hyperparameters etc

LIMIT_INPUT_NAMES = None

BLOCK_SIZE = 3  # how many preceding characters we use as X inputs to predict with
CHARACTER_DIMENSIONS = 6  # how many numbers we use to represent a character
LAYER_1_COUNT_NEURONS = 200

# TRAINING_CYCLES = 100000
TRAINING_CYCLES = 40000
BATCH_SIZE = 256
LEARNING_RATE_1 = 0.6
LEARNING_RATE_2 = 0.05
LEARNING_RATE_TRANSITION_AT_CYCLE = int(TRAINING_CYCLES / 2)

# Misc constants
EDGE_MARKER = "."  # depends on this character not appearing in the names.txt file


# Load names from the file
BASE_PATH = "/Users/dave/Dropbox/Projects/Learning/Neural Networks/Karpathy/neural-network-learning-karpathy"
names_file = f"{BASE_PATH}/resources/names.txt"
names = open(names_file).read().splitlines()
if LIMIT_INPUT_NAMES is not None:
    names = names[:LIMIT_INPUT_NAMES]  # limit data set for development work
print(f"{len(names)} names in input data set.")

# We'll represent characters by integer codes
char_set: Set[str] = set()
for name in names:
    for char in name:
        char_set.add(char)
char_set.add(EDGE_MARKER)
chars = sorted(char_set)
# # Hard-code the char set for now, to perfectly match numbers in the videos.
# chars = [EDGE_MARKER, "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
# assert len(chars) == 27
char_to_code: Dict[str, int] = {char: code for code, char in enumerate(chars)}
code_to_char: Dict[int, str] = {code: char for code, char in enumerate(chars)}


# Create data sets, divided into training, dev, and test
# X inputs are BLOCK_SIZE element vectors, holding codes of the BLOCK_SIZE preceding chars.
# Y output is 1 element vector, HOLDING code of predicted next char

def data_for_names(names_: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[List[int]] = []
    ys: List[int] = []
    for name in names_:
        if len(names) < 10:
            print(name)
        # EDGE_MARKER represents positions before or after the name
        name = EDGE_MARKER * BLOCK_SIZE + name + EDGE_MARKER
        for pos in range(len(name) - BLOCK_SIZE):
            x_string = name[pos:pos+BLOCK_SIZE]   # BLOCK_SIZE characters preceding y_string
            y_string = name[pos+BLOCK_SIZE]       # 1 character
            x = [char_to_code[char] for char in x_string]
            y = char_to_code[y_string]
            xs.append(x)
            ys.append(y)
            if len(names) < 10:
                print(f"{''.join([code_to_char[code] for code in x])} ---> {code_to_char[y]}")
    assert len(xs) == len(ys)
    X_ = torch.tensor(xs)
    Y_ = torch.tensor(ys)
    return X_, Y_


# We divide our names data set up into 3 subsets: 80% for training, 10% for dev testing, and 10% for final test
random.seed(42)
random.shuffle(names)
# count_dev = max(int(len(names) * 0.1), 1)
# count_test = count_dev
# count_training = len(names) - count_dev - count_test
# n1 = count_training
# n2 = count_training + count_dev
n1 = int(0.8*len(names))
n2 = int(0.9*len(names))
X_training, Y_training = data_for_names(names[:n1])
X_dev, Y_dev = data_for_names(names[n1:n2])
X_test, Y_test = data_for_names(names[n2:])
print(f"Data set: training={X_training.shape[0]} dev={X_dev.shape[0]} test={X_test.shape[0]}")


# Initial lookup matrix
# Maps character code one-hot vectors to vectors of CHARACTER_DIMENSIONS size
# Each row N is the vector for character N, so you can directly index as well
generator = torch.Generator().manual_seed(2147483647)
C = torch.randn((len(chars), CHARACTER_DIMENSIONS), generator=generator)

# Layer 1
# Maps the combined character vectors for the characters in the preceding block to vector containing one output
# float per neuron
W1 = torch.randn((CHARACTER_DIMENSIONS * BLOCK_SIZE, LAYER_1_COUNT_NEURONS), generator=generator) * 0.2
b1 = torch.randn(LAYER_1_COUNT_NEURONS, generator=generator) * 0.01

# Layer 2
# Maps layer 1 output to probability vector of size len(chars)
W2 = torch.randn((LAYER_1_COUNT_NEURONS, len(chars)), generator=generator) * 0.1
b2 = torch.randn(len(chars), generator=generator)

# Require grad for our leaf parameters.
# Must be done before the forward pass, in order for the operations we perform to have a grad function
# attached to them
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
print(f"{sum(p.nelement() for p in parameters)} trainable parameters in the model.")


def logits_for_x(X_: torch.Tensor) -> torch.Tensor:
    l1_out_before = C[X_].view(-1, CHARACTER_DIMENSIONS * BLOCK_SIZE) @ W1 + b1
    # plt.hist(l1_out_before.view(-1).tolist(), 50); plt.show()
    l1_out = torch.tanh(l1_out_before)
    # plt.hist(l1_out.view(-1).tolist(), 50); plt.show()
    logits_ = l1_out @ W2 + b2
    return logits_


def forward_pass(X_: torch.Tensor, Y_: torch.Tensor) -> torch.Tensor:
    logits_ = logits_for_x(X_)
    loss_ = F.cross_entropy(logits_, Y_)
    return loss_


def backward_pass(loss_: torch.Tensor, learning_rate: float) -> None:
    for p in parameters:
        p.grad = None
    loss_.backward()
    for p in parameters:
        p.data += -learning_rate * p.grad


# Training loop
print(f"Training for {TRAINING_CYCLES} cycles ", end="")
sys.stdout.flush()
losses = []
for cycle_num in range(TRAINING_CYCLES):
    # Obtain this batch
    batch_indices = torch.randint(0, X_training.shape[0], (BATCH_SIZE,), generator=generator)
    X_batch = X_training[batch_indices]
    Y_batch = Y_training[batch_indices]

    # Forward pass
    loss = forward_pass(X_batch, Y_batch)
    losses.append(loss.item())
    if TRAINING_CYCLES <= 100 or cycle_num < 20:
        print(f"Batch loss: {loss.item()}")

    # Backward pass
    learning_rate = LEARNING_RATE_1 if cycle_num < LEARNING_RATE_TRANSITION_AT_CYCLE else LEARNING_RATE_2
    backward_pass(loss, learning_rate=learning_rate)

    # Progress indicator
    if (cycle_num + 1) % 10000 == 0:
        print(".", end="")
        sys.stdout.flush()
print("")
print("Training complete.")
plt.plot(losses)
plt.show()


# Forward pass with the different slices
print("")
training_loss = forward_pass(X_training, Y_training)
dev_loss = forward_pass(X_dev, Y_dev)
print(f"training loss {training_loss:.6}, dev loss {dev_loss:.6}")

# Generate new names
print("")
generator2 = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(10):
    out_codes = []
    current_chars = [0] * BLOCK_SIZE
    while True:
        logits = logits_for_x(current_chars)
        probs = F.softmax(logits, dim=1)
        next_char = torch.multinomial(probs, num_samples=1, generator=generator2).item()
        current_chars = current_chars[1:] + [next_char]
        if next_char == 0:
            break
        out_codes.append(next_char)
    print(''.join(code_to_char[i] for i in out_codes))

print("")
print("Done")
