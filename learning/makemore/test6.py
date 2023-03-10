# Video 4
# Initial code copied from https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures


# read in all the words
# words = open('names.txt', 'r').read().splitlines()
BASE_PATH = "/Users/dave/Dropbox/Projects/Learning/Neural Networks/Karpathy/neural-network-learning-karpathy"
names_file = f"{BASE_PATH}/resources/names.txt"
words = open(names_file).read().splitlines()
print(len(words))
print(max(len(w) for w in words))
print(words[:8])


# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)


# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%


# ok biolerplate done, now we get to the action:


# utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')


n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 64 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1 # using b1 just for fun, it's useless because of BN
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

# Note: I am initializating many of these parameters in non-standard ways
# because sometimes initializating with e.g. all zeros could mask an incorrect
# implementation of the backward pass.

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
    p.requires_grad = True


batch_size = 32
n = batch_size # a shorter variable also, for convenience
# construct a minibatch
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y


# forward pass, "chunkated" into smaller steps that are possible to backward one at a time

emb = C[Xb] # embed the characters into vectors
embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
# Linear layer 1
hprebn = embcat @ W1 + b1 # hidden layer pre-activation
# BatchNorm layer
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
# Non-linearity
h = torch.tanh(hpreact) # hidden layer
# Linear layer 2
logits = h @ W2 + b2 # output layer
# cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(dim=1, keepdim=True).values
norm_logits = logits - logit_maxes # subtract max for numerical stability
counts = norm_logits.exp()
counts_sum = counts.sum(dim=1, keepdims=True)
counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

# PyTorch backward pass
for p in parameters:
    p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, # afaik there is no cleaner way
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
          bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
          embcat, emb]:
    t.retain_grad()
loss.backward()
print(loss)


# Exercise 1: backprop through the whole thing manually,
# backpropagating through exactly all of the variables
# as they are defined in the forward pass above, one by one

dlogprobs = torch.zeros_like(logprobs).index_put((torch.tensor(range(n)), Yb), torch.tensor(-1.0 / n))
# dlogprobs = torch.zeroes_like(logprobs)   # Kaparthy
# dlogprobs[range(n), Yb] = -1.0 / n
cmp('logprobs', dlogprobs, logprobs)

dprobs = (1.0 / probs) * dlogprobs
cmp('probs', dprobs, probs)

dcounts_sum_inv = (counts * dprobs).sum(dim=1, keepdim=True)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)

dcounts_sum = -(counts_sum ** -2.0) * dcounts_sum_inv
cmp('counts_sum', dcounts_sum, counts_sum)

dcounts = 1.0 * dcounts_sum.repeat(1, vocab_size) + counts_sum_inv * dprobs  # counts used twice, so we add the two contributions
cmp('counts', dcounts, counts)

dnorm_logits = norm_logits.exp() * dcounts
cmp('norm_logits', dnorm_logits, norm_logits)

dlogit_maxes = (-1.0 * dnorm_logits).sum(1, keepdim=True)  # logit_maxes is 32x1, so broadcasts out to subtract from all 27 cols of logits. We have to sum up all these contributions
cmp('logit_maxes', dlogit_maxes, logit_maxes)

_, indices = logits.max(dim=1, keepdim=True)
ones_max_logits = torch.zeros_like(logits).scatter(1, indices, 1.0)
dlogits = ones_max_logits * dlogit_maxes + 1.0 * dnorm_logits  # should be 1 for each element that is the max in its row, 0 for others
cmp('logits', dlogits, logits)

# logits (32x27) = h (32x64) @ W2 (64x27) + b(27)
# dh/dloss (32x64) = dlogits/dloss (32x27) @ transpose(W2) (27x64)
dh = dlogits @ W2.transpose(0, 1)
cmp('h', dh, h)

# logits (32x27) = h (32x64) @ W2 (64x27) + b(27)
# dW2/dloss (64x27) = transpose(h) (64x32) @ dlogits/dloss (32x27)
dW2 = h.transpose(0, 1) @ dlogits
cmp('W2', dW2, W2)

db2 = dlogits.sum(dim=0, keepdim=True)
cmp('b2', db2, b2)

dhpreact = (torch.ones_like(h) - h ** 2) * dh
# dhpreact = (1.0 - h**2) * dh    # Karpathy
cmp('hpreact', dhpreact, hpreact)

dbngain = (bnraw * dhpreact).sum(dim=0, keepdim=True)
cmp('bngain', dbngain, bngain)

dbnbias = dhpreact.sum(dim=0, keepdim=True)
cmp('bnbias', dbnbias, bnbias)

dbnraw = bngain * dhpreact
cmp('bnraw', dbnraw, bnraw)

dbnvar_inv = (dbnraw * bndiff).sum(0, keepdim=True)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)

dbnvar = -0.5 * (bnvar + 1e-5) ** -1.5 * dbnvar_inv
cmp('bnvar', dbnvar, bnvar)

dbndiff2 = (1.0 / (n - 1.0)) * dbnvar.repeat(n, 1)
cmp('bndiff2', dbndiff2, bndiff2)

dbndiff = 2.0 * bndiff * dbndiff2 + bnvar_inv * dbnraw
cmp('bndiff', dbndiff, bndiff)

dbnmeani = -1.0 * dbndiff.sum(dim=0, keepdim=True)
cmp('bnmeani', dbnmeani, bnmeani)

dhprebn = (1.0 / n) * dbnmeani.repeat(n, 1) + 1.0 * dbndiff
cmp('hprebn', dhprebn, hprebn)

# hprebn (32x64) = embcat (32x30) @ W1 (30x64) + b1
dembcat = dhprebn @ W1.transpose(0, 1)
cmp('embcat', dembcat, embcat)

# hprebn (32x64) = embcat (32x30) @ W1 (30x64) + b1
dW1 = embcat.transpose(0, 1) @ dhprebn
cmp('W1', dW1, W1)

db1 = dhprebn.sum(dim=0, keepdim=False)
cmp('b1', db1, b1)

demb = dembcat.view(emb.shape[0], emb.shape[1], emb.shape[2])
cmp('emb', demb, emb)

# emb = C[Xb]
# Takes Xb (32x3 char values 0-26) and C (27x10 char vector elements), maps to (32x3x10)
# Influence of each element of C is just going to be the sum of all the dL/d_emb the corresponding positions in demb
# Algo:
#   - initialize dC to zeroes like C (27x10)
#   - loop through all elements of Xb: x 0-31, y 0-2:
#   -    obtain its element value 0-26 char:
#   -    loop through that row in C: z 0-9:
#   -        dC[char, z] += demb[x, y, z]
dC = torch.zeros_like(C)
for xB_row_index, xB_row_tensor in enumerate(Xb):  # loop thru training data points
    for xB_col_index, xB_col_tensor in enumerate(xB_row_tensor):   # loop through previous characters
        char_code = xB_col_tensor.item()
        for C_col_index, _ in enumerate(C[char_code]):
            dC[char_code, C_col_index] += demb[xB_row_index, xB_col_index, C_col_index]
cmp('C', dC, C)
# Karpathy - much cleaner and easier to read
# dC = torch.zeros_like(C)
# for k in range(Xb.shape[0]):
#     for j in range(Xb.shape[1]):
#         ix = Xb[k, j]
#         dC[ix] += demb[k, j]


# Exercise 2: backprop through cross_entropy but all in one go
# to complete this challenge look at the mathematical expression of the loss,
# take the derivative, simplify the expression, and just write it out

# Cross entropy:
# e^x on logits - row max value -> "counts"
# normalize by dividing each element by row sum -> "probs"
# take average of prob for expected char column, negate -> "loss"
#    this is all elementwise operations and size stays 32x27 until we squash to 32x1 at the end



# Exercise 3: backprop through batchnorm but all in one go
# to complete this challenge look at the mathematical expression of the output of batchnorm,
# take the derivative w.r.t. its input, simplify the expression, and just write it out


# Exercise 4: putting it all together!
# Train the MLP neural net with your own backward pass


print("Done")