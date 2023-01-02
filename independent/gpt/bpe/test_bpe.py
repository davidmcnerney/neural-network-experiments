from textwrap import dedent
from typing import Dict

from independent.gpt.bpe import builder
from independent.gpt.bpe import text_processing


def test_pretokenize():
    string = "Big jug, dig dug."
    expected_words = [
        "Big",
        " jug",
        ",",
        " dig",
        " dug",
        "."
    ]

    assert text_processing.pretokenize(string) == expected_words


def test_to_unicode_bytes():
    string = "Big jug, dig dug."
    expected = "BigĠjug,ĠdigĠdug."
    assert text_processing.to_unicode_bytes(string) == expected


def test_build_vocab():
    training_text = """big"""

    expected_vocab = {
        256: "ig",
        257: "big",
    }
    expected_vocab.update(_byte_vocab())

    expected_merges = [
        ("i", "g", "ig"),
        ("b", "ig", "big"),
    ]

    actual_vocab, actual_merges = builder.build_vocabulary_and_merge_list(
        training_text=training_text,
        count_merges=5,
    )
    assert actual_vocab == expected_vocab
    assert actual_merges == expected_merges


def _byte_vocab() -> Dict[int, str]:
    return dict(text_processing.BYTE_TO_UNICODE_REPRESENTATION)
