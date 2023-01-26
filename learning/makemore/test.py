from typing import Dict, Tuple

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
# print(min(len(w) for w in names))
# len_2_names = [name for name in names if len(name) == 2]
# print(f"words with length 2: {len(len_2_names)}")
# print(sorted(len_2_names))

counts: Dict[Tuple[str, str], int] = {}
for name in names:
    name = "." + name + "."
    for ch1, ch2 in zip(name, name[1:]):
        counts[(ch1, ch2)] = counts.get((ch1, ch2), 0) + 1
for transition, count in reversed(sorted(counts.items(), key=lambda transition_and_count: transition_and_count[1])):
    print(f"{transition[0]} -> {transition[1]} : {count}")
