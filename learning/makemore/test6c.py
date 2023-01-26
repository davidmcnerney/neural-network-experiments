import torch

list_of_lists = [[3.0, 2.0, 1.0], [4.0, 5.0, 6.0], [9.0, 7.0, 8.0]]
t1 = torch.tensor(list_of_lists)
for row_index, row_tensor in enumerate(t1):
    for col_index, col_tensor in enumerate(row_tensor):
        print(f"{row_index},{col_index}: {col_tensor.item()}")

# print(t1[2, 1])
