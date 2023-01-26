

# Below is a more broken-apart version of the above logits expression
# C_out = C[X]
# print(f"C_out: {C_out.shape}")
#
# # C_out is a 3 dimensional tensor of size len(X) x BLOCK_SIZE x CHARACTER_DIMENSIONS, for example
# # 30 x 3 x 2. We need a 2 dimensional tensor of size len(X) x BLOCK_SIZE * CHARACTER_DIMENSIONS,
# # in this example 30 x 6. We want to feed single vectors with all the character vectors just concatenated end-to-end
# # into the first layer of the neural network. The view method does this efficiently for us.
# C_out_flattened = C_out.view(-1, CHARACTER_DIMENSIONS * BLOCK_SIZE)
# print(f"C_out_flattened: {C_out_flattened.shape}")
# # b1 is broadcast to add its weights to every row of the output
# l1_out = torch.tanh(C_out_flattened @ W1 + b1)
# print(f"l1_out: {l1_out.shape}")
#
# l2_out = l1_out @ W2 + b2
# print(f"l2_out: {l2_out.shape}")
# logits = l2_out

# Below is the explicit equivalent of cross_entropy
# # Softmax
# counts = logits.exp()
# sums = counts.sum(dim=1, keepdims=True)
# print(f"sums: {sums.shape}")
# probs = counts / sums
# print(f"probs: {probs.shape}")
# # Loss
# ar = torch.arange(Y.size(dim=0))
# relevant = probs[ar, Y]
# loss = -relevant.log().mean()
