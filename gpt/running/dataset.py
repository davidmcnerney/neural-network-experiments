from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from gpt.bpe import input_output
from gpt.bpe import tokenizer


class InMemoryTokenDataset(Dataset):
    def __init__(
            self,
            tokens: List[int],
            block_size: int,
            device: torch.device,
    ):
        self.tokens = tokens
        self.block_size = block_size
        self.device = device

    def __getitem__(self, item):
        x = self.tokens[item:(item + self.block_size)]
        y = self.tokens[(item + 1):(item + 1 + self.block_size)]
        return torch.tensor(x, device=self.device), torch.tensor(y, device=self.device)

    def __len__(self):
        # For each sample that we return, we need an X sequence and a Y sequence. Each element of Y is just
        # the index + 1 position token from X. For example:
        #       len(tokens) 4
        #       block_size  3
        #       available samples: only 1
        #           X will be indices 0, 1 and 2
        #           Y will be indices 1, 2 and 3
        count_available_blocks = len(self.tokens) - self.block_size + 1 - 1
        return max(count_available_blocks, 0)

    @classmethod
    def load(
            cls,
            filename: str,
            vocab_filename: str,
            merge_filename: str,
            block_size: int,
            device: torch.device,
    ) -> Tuple["InMemoryTokenDataset", "InMemoryTokenDataset"]:
        """
        Returns training (90%) and validation (10%) datasets.
        """
        with open(filename) as file:
            input_text = file.read()
        tokens = cls._tokenize(input_text, vocab_filename, merge_filename)
        count_training_tokens, count_validation_tokens = cls._divide_tokens(len(tokens))
        return (
            cls(tokens=tokens[:count_training_tokens], block_size=block_size, device=device),
            cls(tokens=tokens[-count_validation_tokens:], block_size=block_size, device=device),
        )

    @staticmethod
    def _tokenize(
            input_text: str,
            vocab_filename: str,
            merge_filename: str,
    ) -> List[int]:
        vocab = input_output.load_vocabulary_from_file(vocab_filename)
        merges = input_output.load_merge_list_from_file(merge_filename)
        return tokenizer.tokenize(input_text, vocab, merges)

    @staticmethod
    def _divide_tokens(count_tokens: int) -> Tuple[int, int]:
        """
        Divides tokens into training and validation portions.
        This algorithm doesn't take block size into account, and maybe should.
        """
        if count_tokens == 0:
            return 0, 0
        elif count_tokens == 1:
            return 1, 0
        elif count_tokens <= 10:
            return count_tokens - 1, 1
        else:
            count_validation_tokens = round(0.1 * count_tokens)
            return count_tokens - count_validation_tokens, count_validation_tokens
