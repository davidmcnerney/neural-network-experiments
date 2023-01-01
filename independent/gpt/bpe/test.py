from textwrap import dedent

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
    training_text = dedent("""
        Big jug, dig dug.
    """)

    expected_vocab = {
        0: "B",
        # etc
    }

    expected_merges = [
        (("B", "i"), "Bi")
        # etc
    ]

    assert builder.build_vocab(training_text) == (expected_vocab, expected_merges)
