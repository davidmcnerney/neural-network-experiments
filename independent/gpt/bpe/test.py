from textwrap import dedent

from independent.gpt.bpe.builder import build_vocab
from independent.gpt.bpe.text_processing import pretokenize


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

    assert pretokenize(string) == expected_words


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

    assert build_vocab(training_text) == (expected_vocab, expected_merges)
