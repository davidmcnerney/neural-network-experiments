from typing import List

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
        return self.tokens[item:item + self.block_size]

    def __len__(self):
        return len(self.tokens) - self.block_size + 1

    @staticmethod
    def _tokenize(
            input_text: str,
            vocab_filename: str,
            merge_filename: str,
    ) -> List[int]:
        vocab = input_output.load_vocabulary_from_file(vocab_filename)
        merges = input_output.load_merge_list_from_file(merge_filename)
        return tokenizer.tokenize(input_text, vocab, merges)
