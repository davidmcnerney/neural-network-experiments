from typing import Dict, List

import regex


#
# Pretokenization
#


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
    """
    Breaks a string down into our initial larger tokens that mostly correspond to
    words. See unit tests for example.
    """
    return regex.findall(PRETOKENIZE_REGEX, string)


#
# Bytes to Unicode
#


def _create_byte_to_unicode_representation() -> Dict[int, str]:
    # These are bytes that are readily readable when rendered as characters,
    # such as 65 -> "A".
    readable_bytes = list(range(ord("!"), ord("~") + 1)) \
        + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))

    # Starting at 256, we have a series of other characters that display well.
    # We use those to represent the bytes from 0-255 which are not readily
    # readable.
    next_alternate_character = 256

    byte_to_unicode: Dict[int, str] = {}
    for byte in range(256):
        if byte in readable_bytes:
            byte_to_unicode[byte] = chr(byte)
        else:
            byte_to_unicode[byte] = chr(next_alternate_character)
            next_alternate_character += 1

    return byte_to_unicode


BYTE_TO_UNICODE_REPRESENTATION = _create_byte_to_unicode_representation()


def to_unicode_bytes(string: str) -> str:
    """
    Converts a Unicode string into a another Unicode string that represents its bytes.
    Simple characters like ! or A will look the same. Characters that aren't so readable
    like spaces will change to other characters that are easier to read. Unicode characters
    will change to other characters that represent their bytes. See unit tests for example.
    """
    string_in_bytes = string.encode("utf-8")
    return "".join([BYTE_TO_UNICODE_REPRESENTATION[byte] for byte in string_in_bytes])
