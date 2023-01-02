from typing import List

from independent.gpt.bpe import text_processing
from independent.gpt.bpe import type_definitions


def coarse_tokenize(text: str) -> List[str]:
    # TODO: cache
    return [
        text_processing.to_unicode_bytes(token)
        for token in text_processing.pretokenize(text)
    ]


def fine_tokenize(string: str, merge_list: type_definitions.MergeList) -> List[str]:
    # Start by breaking the string into 1 byte tokens
    tokens = list(string)

    # Work through the string, checking each pair of tokens in turn for presence in the merge list
    while True:
        out_tokens: List[str] = []
        start_index = 0
        while start_index < len(tokens):
            if start_index < len(tokens) - 1:
                first = tokens[start_index]
                second = tokens[start_index + 1]
                if (first, second) in merge_list:
                    out_tokens.append(merge_list[first, second])
                    start_index += 2
                    continue
            out_tokens.append(tokens[start_index])
            start_index += 1

        if out_tokens == tokens:
            return out_tokens
        tokens = out_tokens  # loop around to continue merging larger and larger tokens
