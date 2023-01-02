from textwrap import dedent
from typing import Dict, List

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
    training_text = """Big jug, dig dug."""

    vocab, merges = builder.build_vocabulary_and_merge_list(
        training_text=training_text,
        count_merges=1000,
    )

    # print("\n\n")
    # builder.summarize_vocab(builder.remove_base_byte_vocab(vocab))
    # builder.summarize_merges(merges)
    # assert False

    expected_vocab = {
        256: "Ġd",
        257: "ug",
        258: "ig",
        259: "Ġdug",
        260: "Ġdig",
        261: "jug",
        262: "Ġjug",
        263: "Big",
    }
    assert builder.remove_base_byte_vocab(vocab) == expected_vocab

    expected_merges = dedent("""
        Ġ d
        u g
        i g
        Ġd ug
        Ġd ig
        j ug
        Ġ jug
        B ig
    """)
    assert "\n" + builder.serialize_merges(merges) == expected_merges


def _byte_vocab() -> Dict[int, str]:
    return dict(text_processing.BYTE_TO_UNICODE_REPRESENTATION)
