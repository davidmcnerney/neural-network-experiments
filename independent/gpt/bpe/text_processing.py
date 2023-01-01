from typing import List

import regex


PRETOKENIZE_REGEX = regex.compile(
    r"""
    's                          # various contraction suffixes
    |'t
    |'re
    |'ve
    |'m
    |'ll
    |'d
    |\ ?\p{L}+                  # space + Unicode letters
    |\ ?\p{N}+                  # space + Unicode digits
    |\ ?[^\s\p{L}\p{N}]+        # optional space + not whitespace, Unicode letters or digits
    |\s+(?!\S)                  # whitespace not followed by non-whitespace
    |\s+                        # full trailing whitespace
    """,
    regex.VERBOSE,
)


def pretokenize(string: str) -> List[str]:
    return regex.findall(PRETOKENIZE_REGEX, string)
