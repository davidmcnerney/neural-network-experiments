from typing import List

import torch
from torch.utils.data import Dataset

from independent.gpt.bpe import input_output
from independent.gpt.bpe import tokenizer


class TinyShakespeareDataset(Dataset):
    def __init__(
            self,
            filename: str,
            vocab_filename: str,
            merge_filename: str,
            block_size: int,
    ):
        self.block_size = block_size

        with open(filename) as file:
            input_text = file.read()
        self.tokens = self._tokenize(input_text, vocab_filename, merge_filename)

    def __getitem__(self, item):
        x = self.tokens[item:(item + self.block_size)]
        y = self.tokens[(item + 1):(item + 1 + self.block_size)]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        # For each sample that we return, we need an X sequence and a Y sequence. Each element of Y is just
        # the index + 1 position token from X. For example:
        #       len(tokens) 4
        #       block_size  3
        #       available samples: only 1
        #           X will be indices 0, 1 and 2
        #           Y will be indices 1, 2 and 3
        return len(self.tokens) - self.block_size + 1 - 1

    @staticmethod
    def _tokenize(
            input_text: str,
            vocab_filename: str,
            merge_filename: str,
    ) -> List[int]:
        vocab = input_output.load_vocabulary_from_file(vocab_filename)
        merges = input_output.load_merge_list_from_file(merge_filename)
        return tokenizer.tokenize(input_text, vocab, merges)
