# import torch
#
# x = torch.rand(5, 3)
# print(x)

BASE_PATH = "/Users/dave/Dropbox/Projects/Learning/Neural Networks/Karpathy/neural-network-learning-karpathy"

names_file = f"{BASE_PATH}/resources/names.txt"
names = open(names_file).read().splitlines()
print(f"len: {len(names)}")
# sorted_names = sorted(names)
# print(sorted_names[1000:1150])
print(min(len(w) for w in names))
len_2_names = [name for name in names if len(name) == 2]
print(f"words with length 2: {len(len_2_names)}")
print(sorted(len_2_names))
