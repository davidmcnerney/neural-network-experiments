import random
import sys
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.nn import functional as F


# Video 3


# Hyperparameters etc

LIMIT_INPUT_NAMES = None

BLOCK_SIZE = 3  # how many preceding characters we use as X inputs to predict with
CHARACTER_DIMENSIONS = 10  # how many numbers we use to represent a character
LAYER_COUNT_NEURONS = 100

TRAINING_CYCLES = 40000
BATCH_SIZE = 32
LEARNING_RATE_1 = 0.1
LEARNING_RATE_2 = 0.01
LEARNING_RATE_TRANSITION_AT_CYCLE = 10000

# Misc constants
EDGE_MARKER = "."  # depends on this character not appearing in the names.txt file


generator = torch.Generator().manual_seed(2147483647)


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.weight = torch.randn((in_features, out_features), generator=generator)
        if bias:
            self.bias = torch.zeros(out_features)
        self.out: Optional[torch.tensor] = None

    def __call__(self, x) -> torch.Tensor:
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self) -> List[torch.Tensor]:
        return_value = [self.weight]
        if self.bias is not None:
            return_value.append(self.bias)
        return return_value


class BatchNorm1d:
    def __init__(self, dim, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps            # small constant to prevent divide by zero errors
        self.momentum = momentum  # how quickly running_mean and running_var update

        self.training = True

        # Trainable parameters that tune the normalization
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        # Initialize buffers: we capture average mean and variance while training, for use in inference later
        self.running_mean = torch.zeros(dim)
        self.running_variance = torch.ones(dim)

    def __call__(self, x) -> torch.Tensor:
        if self.training:
            # Compute batch mean and variance during training
            mean = x.mean(dim=0, keepdim=True)
            variance = x.var(dim=0, keepdim=True)
        else:
            # Use previously captured average mean and variance in evaluation mode
            mean = self.running_mean
            variance = self.running_variance

        x_normalized = (x - mean) / torch.sqrt(variance + self.eps)
        self.out = self.gamma * x_normalized + self.beta

        # Update buffers: our running mean and variance
        if self.training:
            with torch.no_grad():
                self.running_mean = (1.0 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_variance = (1.0 - self.momentum) * self.running_variance + self.momentum * variance

        return self.out

    def parameters(self) -> List[torch.Tensor]:
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x) -> torch.Tensor:
        self.out = torch.tanh(x)
        return self.out

    def parameters(self) -> List[torch.Tensor]:
        return []


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
charset_size = len(chars)
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


# Build the network

# Initial lookup matrix
# Maps character code one-hot vectors to vectors of CHARACTER_DIMENSIONS size
# Each row N is the vector for character N, so you can directly index as well
C = torch.randn((charset_size, CHARACTER_DIMENSIONS),                          generator=generator)

# Layers
layers = [
    Linear(CHARACTER_DIMENSIONS * BLOCK_SIZE, LAYER_COUNT_NEURONS), Tanh(),
    Linear(              LAYER_COUNT_NEURONS, LAYER_COUNT_NEURONS), Tanh(),
    Linear(              LAYER_COUNT_NEURONS, LAYER_COUNT_NEURONS), Tanh(),
    Linear(              LAYER_COUNT_NEURONS, LAYER_COUNT_NEURONS), Tanh(),
    Linear(              LAYER_COUNT_NEURONS, LAYER_COUNT_NEURONS), Tanh(),
    Linear(              LAYER_COUNT_NEURONS, LAYER_COUNT_NEURONS), Tanh(),
    Linear(              LAYER_COUNT_NEURONS, charset_size),
]

with torch.no_grad():  # don't understand why do this here
    # Apple gain to all layers but the last
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5.0 / 3.0

    # Make last layer less "confident"
    layers[-1].weight *= 0.1


# Require grad for our leaf parameters.
# Must be done before the forward pass, in order for the operations we perform to have a grad function
# attached to them
parameters = [C] + [p for l in layers for p in l.parameters()]
for p in parameters:
    p.requires_grad = True
print(f"{sum(p.nelement() for p in parameters)} trainable parameters in the model.")


def logits_for_x(X_: torch.Tensor) -> torch.Tensor:
    input_vectors = C[X_]
    x = input_vectors.view(-1, CHARACTER_DIMENSIONS * BLOCK_SIZE)
    # plt.hist(x.view(-1).tolist(), 50); plt.show()
    for layer in layers:
        x = layer(x)
        # plt.hist(x.view(-1).tolist(), 50); plt.show()
    return x


def forward_pass(X_: torch.Tensor, Y_: torch.Tensor) -> torch.Tensor:
    logits_ = logits_for_x(X_)
    loss_ = F.cross_entropy(logits_, Y_)
    return loss_


def backward_pass(loss_: torch.Tensor, learning_rate: float) -> None:
    # Don't understand what this is for, but supposed to be necessary only for dev work
    for layer_ in layers:
        layer_.out.retain_grad()

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
    # if TRAINING_CYCLES <= 100 or cycle_num < 20:
    #     print(f"Batch loss: {loss.item()}")

    # Backward pass
    learning_rate = LEARNING_RATE_1 if cycle_num < LEARNING_RATE_TRANSITION_AT_CYCLE else LEARNING_RATE_2
    backward_pass(loss, learning_rate=learning_rate)

    # Progress indicator
    if (cycle_num + 1) % 10000 == 0:
        print(".", end="")
        sys.stdout.flush()
print("")
print("Training complete.")
# plt.plot(losses); plt.show()


# Forward pass with the different slices
print("")
training_loss = forward_pass(X_training, Y_training)
dev_loss = forward_pass(X_dev, Y_dev)
print(f"training loss {training_loss:.6}, dev loss {dev_loss:.6}")


# Sampling - generate new names
print("")
generator2 = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(10):
    out_codes = []
    current_chars = [0] * BLOCK_SIZE
    while True:
        logits = logits_for_x(current_chars)
        probs = F.softmax(logits, dim=1)
        next_char = torch.multinomial(probs, num_samples=1, generator=generator2).item()
        if next_char == 0:
            break
        out_codes.append(next_char)
        current_chars = current_chars[1:] + [next_char]
    print(''.join(code_to_char[i] for i in out_codes))

print("")
print("Done")
