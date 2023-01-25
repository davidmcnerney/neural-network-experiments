from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from independent.gpt.bpe import input_output
from independent.gpt.bpe import tokenizer


class InMemoryTokenDataset(Dataset):
    def __init__(
            self,
            blocks: List[List[int]],  # to contain block_size + 1 elements, so we can source X and Y token lists both
            block_size: int,
            device: torch.device,
    ):
        self.blocks = blocks
        self.block_size = block_size
        self.device = device

    def __getitem__(self, item):
        block = self.blocks[item]
        x = block[:self.block_size]
        y = block[1:]
        return torch.tensor(x, device=self.device), torch.tensor(y, device=self.device)

    def __len__(self):
        return len(self.blocks)

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
        blocks = cls._divide_tokens_into_blocks(tokens, block_size)
        count_training_blocks, count_validation_blocks = cls._divide_blocks(len(blocks))
        return (
            cls(blocks=blocks[:count_training_blocks], block_size=block_size, device=device),
            cls(blocks=blocks[-count_validation_blocks:], block_size=block_size, device=device),
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
    def _divide_tokens_into_blocks(tokens: List[int], block_size: int) -> List[List[int]]:
        count_blocks = (len(tokens) - 1) // block_size  # extra 1 to accommodate Y values
        blocks: List[List[int]] = []
        for block_num in range(count_blocks):
            start_index = block_num * block_size
            block = tokens[start_index:start_index + block_size + 1]  # extra 1 to accommodate Y values
            blocks.append(block)
        return blocks

    @staticmethod
    def _divide_blocks(count_blocks: int) -> Tuple[int, int]:
        """
        Divides a block count into training and validation portions.
        """
        if count_blocks == 0:
            return 0, 0
        elif count_blocks == 1:
            return 1, 0
        elif count_blocks <= 10:
            return count_blocks - 1, 1
        else:
            count_validation_blocks = round(0.1 * count_blocks)
            return count_blocks - count_validation_blocks, count_validation_blocks
