import random
import sys
from typing import cast, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F


# Video 5


# Hyperparameters etc

LIMIT_INPUT_NAMES = None

BLOCK_SIZE = 8  # how many preceding characters we use as X inputs to predict with
CHARACTER_DIMENSIONS = 24  # how many numbers we use to represent a character
LAYER_COUNT_NEURONS = 128

TRAINING_CYCLES = 200000
BATCH_SIZE = 32
LEARNING_RATE_1 = 0.1
LEARNING_RATE_2 = 0.01
LEARNING_RATE_TRANSITION_AT_CYCLE = 150000

# Misc constants
EDGE_MARKER = "."  # depends on this character not appearing in the names.txt file


torch.manual_seed(2147483647)


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
n1 = int(0.8*len(names))
n2 = int(0.9*len(names))
X_training, Y_training = data_for_names(names[:n1])
X_dev, Y_dev = data_for_names(names[n1:n2])
X_test, Y_test = data_for_names(names[n2:])
print(f"Data set: training={X_training.shape[0]} dev={X_dev.shape[0]} test={X_test.shape[0]}")


# Build the model

model = nn.Sequential(
    nn.Embedding(charset_size, CHARACTER_DIMENSIONS),
    nn.Flatten(), nn.Linear(CHARACTER_DIMENSIONS * BLOCK_SIZE, LAYER_COUNT_NEURONS, bias=False), nn.BatchNorm1d(LAYER_COUNT_NEURONS), nn.Tanh(),
    nn.Linear(LAYER_COUNT_NEURONS, charset_size),
)

with torch.no_grad():  # so that we don't add this operation to the computational graph?
    # Make last layer less "confident"
    last_module = cast(nn.Linear, list(model.children())[-1])
    last_module.weight *= 0.1


# Require grad for our leaf parameters.
# Must be done before the forward pass, in order for the operations we perform to have a grad function
# attached to them
model.requires_grad_(True)

print(f"{sum(p.nelement() for p in model.parameters())} trainable parameters in the model.")


def forward_pass(X_: torch.Tensor, Y_: torch.Tensor) -> torch.Tensor:
    logits_ = model(X_)
    loss_ = F.cross_entropy(logits_, Y_)
    return loss_


def backward_pass(loss_: torch.Tensor, learning_rate: float) -> None:
    # for layer_ in model.layers:
    #     layer_.out.retain_grad()

    for p in model.parameters():
        p.grad = None
    loss_.backward()
    for p in model.parameters():
        p.data += -learning_rate * p.grad


# Training loop
print(f"Training for {TRAINING_CYCLES} cycles ", end="")
sys.stdout.flush()
losses: List[float] = []
update_to_data_ratios: List[List[float]] = []
for cycle_num in range(TRAINING_CYCLES):
    # Obtain this batch
    batch_indices = torch.randint(0, X_training.shape[0], (BATCH_SIZE,))
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

    # Track update to data ratio
    with torch.no_grad():
        update_to_data_ratios.append(
            [((learning_rate * p.grad).std() / p.data.std()).log10().item() for p in model.parameters()]
        )

    # Progress indicator
    if (cycle_num + 1) % 10000 == 0:
        print(".", end="")
        sys.stdout.flush()

    # Testing
    # break
print("")
print("Training complete.")


# Plots and reports

# Losses
chunked_losses = torch.tensor(losses).view(-1, 1000).mean(dim=1, keepdim=False).tolist()
plt.figure(figsize=(25, 7))
plt.title("Loss")
plt.plot(chunked_losses)
plt.show()

# # Activations
# plt.figure(figsize=(100, 15))
# legends = []
# print("")
# print("Activations:")
# for i, layer in enumerate(model.layers[:-1]):
#     if isinstance(layer, Tanh):
#         t = layer.out
#         saturation = (t.abs() > 0.97).float().mean().item() * 100.0
#         print(f"layer {i} ({layer.__class__.__name__}): mean {t.mean():.2} std {t.std():.2} sat {round(saturation)}%")
#         hy, hx = torch.histogram(t, density=True)
#         hx = hx[:-1]   # hx is the bin edges, so it has one more element than hy
#         plt.plot(hx.detach(), hy.detach())  # not sure why it's important to call detach() here, I guess to avoid extending the computation graph that Pytorch maintains?
#         legends.append(f"layer {i} ({layer.__class__.__name__})")
# plt.legend(legends)
# plt.title("Activation Distribution")
# plt.show()
#
# # Gradient
# plt.figure(figsize=(100, 15))
# legends = []
# print("")
# print("Gradients:")
# for i, layer in enumerate(model.layers[:-1]):
#     if isinstance(layer, Linear):
#         t = layer.out.grad
#         print(f"layer {i} ({layer.__class__.__name__}): mean {t.mean():.2} std {t.std():.2}")
#         hy, hx = torch.histogram(t, density=True)
#         hx = hx[:-1]   # hx is the bin edges, so it has one more element than hy
#         plt.plot(hx.detach(), hy.detach())  # not sure why it's important to call detach() here, I guess to avoid extending the computation graph that Pytorch maintains?
#         legends.append(f"layer {i} ({layer.__class__.__name__})")
# plt.legend(legends)
# plt.title("Gradient Distribution")
# plt.show()
#
# # Weight gradient distribution
# plt.figure(figsize=(100, 15))
# legends = []
# print("")
# print("Weight Gradients:")
# for i, parameter in enumerate(model.parameters()):
#     if parameter.ndim == 2:
#         t = parameter.grad
#         print(f"parameter {i} weight {tuple(parameter.shape)}  grad mean {t.mean():.2} std {t.std():.2} ratio {t.std() / parameter.std():0.2}")
#         hy, hx = torch.histogram(t, density=True)
#         hx = hx[:-1]   # hx is the bin edges, so it has one more element than hy
#         plt.plot(hx.detach(), hy.detach())  # not sure why it's important to call detach() here, I guess to avoid extending the computation graph that Pytorch maintains?
#         legends.append(f"parameter {i}")
# plt.legend(legends)
# plt.title("Weight Gradient Distribution")
# plt.show()
#
#
# # Update-to-data ratio by parameter
# plt.figure(figsize=(100, 15))
# legends = []
# print("")
# for i, parameter in enumerate(model.parameters()):
#     if parameter.ndim == 2:
#         plt.plot([point[i] for point in update_to_data_ratios])
#         legends.append(f"parameter {i}")
# plt.plot([0, len(update_to_data_ratios)], [-3, -3], "k")
# plt.legend(legends)
# plt.title("Update to Data Ratio")
# plt.show()


# Turn off training mode in the layers
model.train(mode=False)

# Forward pass with the different slices
print("")
training_loss = forward_pass(X_training, Y_training)
dev_loss = forward_pass(X_dev, Y_dev)
print(f"training loss {training_loss:.6}, dev loss {dev_loss:.6}")


# Sampling - generate new names
print("")
for _ in range(10):
    out_codes = []
    current_chars = [0] * BLOCK_SIZE
    while True:
        logits = model(torch.tensor([current_chars]))
        probs = F.softmax(logits, dim=1)
        next_char = torch.multinomial(probs, num_samples=1).item()
        if next_char == 0:
            break
        out_codes.append(next_char)
        current_chars = current_chars[1:] + [next_char]
    print(''.join(code_to_char[i] for i in out_codes))

print("")
print("Done")
