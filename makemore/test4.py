from typing import Dict, List, Optional, Set

import torch
from torch.nn import functional as F

# Video 2: more sophisticated neural net to predict next char of a name, based on BLOCK_SIZE preceding chars

# Hyperparameters etc
BLOCK_SIZE = 3  # how many preceding characters we use as X inputs to predict with

# Misc constants
EDGE_MARKER = "."  # depends on this character not appearing in the names

# Load names from the file
BASE_PATH = "/Users/dave/Dropbox/Projects/Learning/Neural Networks/Karpathy/neural-network-learning-karpathy"
names_file = f"{BASE_PATH}/resources/names.txt"
names = open(names_file).read().splitlines()
names = names[:5]  # limit data set for dev work
print(f"{len(names)} names.")

# We'll represent characters by integer codes
char_set: Set[str] = set()
for name in names:
    for char in name:
        char_set.add(char)
char_set.add(EDGE_MARKER)
chars = sorted(char_set)
char_to_code: Dict[str, int] = {char: code for code, char in enumerate(chars)}
code_to_char: Dict[int, str] = {code: char for code, char in enumerate(chars)}

# Create data set
# X inputs are BLOCK_SIZE element vectors, holding codes of the BLOCK_SIZE preceding chars.
# Y output is 1 element vector, HOLDING code of predicted next char
X: List[List[int]] = []
Y: List[int] = []
for name in names:
    # print(name)
    # EDGE_MARKER represents positions before or after the name
    name = EDGE_MARKER * BLOCK_SIZE + name + EDGE_MARKER
    for pos in range(len(name) - BLOCK_SIZE):
        x_string = name[pos:pos+BLOCK_SIZE]   # BLOCK_SIZE characters preceding y_string
        y_string = name[pos+BLOCK_SIZE]       # 1 character
        x = [char_to_code[char] for char in x_string]
        y = char_to_code[y_string]
        X.append(x)
        Y.append(y)
        # print(f"{''.join([code_to_char[code] for code in x])} ---> {code_to_char[y]}")

# Divide into training, dev, and test data sets
# We divide our names data set up into 3 subsets: 80% for training, 10% for dev testing, and 10% for final test
count_training = round(len(X) * 0.8)
count_dev = round(len(X) * 0.1)
count_test = len(X) - count_dev - count_training
X_training = X[:count_training]
Y_training = Y[:count_training]
X_dev = X[len(X_training):len(X_training) + count_dev]
Y_dev = Y[len(Y_training):len(Y_training) + count_dev]
X_test = X[len(X_training) + len(X_dev):]
Y_test = Y[len(Y_training) + len(Y_dev):]
assert len(X_training) + len(X_dev) + len(X_test) == len(X)
assert len(Y_training) + len(Y_dev) + len(Y_test) == len(Y)
assert len(X_training) == len(Y_training)
assert len(X_dev) == len(Y_dev)
print(f"training: {len(X_training)} dev: {len(X_dev)} test: {len(X_test)}")

print("Done")
