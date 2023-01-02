from textwrap import dedent
from typing import Dict, List

from independent.gpt.bpe import builder
from independent.gpt.bpe import text_processing
from independent.gpt.bpe import tokenizer


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


def test_tokenize():
    text = """Big jug, dig dug."""
    vocab_index_to_token = {
        256: "Ġd",
        257: "ug",
        258: "ig",
        259: "Ġdug",
        260: "Ġdig",
        261: "jug",
        262: "Ġjug",
        263: "Big",
    }
    vocab_index_to_token.update(text_processing.BYTE_TO_UNICODE_REPRESENTATION)
    vocab_token_to_index = builder.invert_vocabulary(vocab_index_to_token)
    merge_list = {
        ("Ġ", "d"): "Ġd",
        ("u", "g"): "ug",
        ("i", "g"): "ig",
        ("Ġd", "ug"): "Ġdug",
        ("Ġd", "ig"): "Ġdig",
        ("j", "ug"): "jug",
        ("Ġ", "jug"): "Ġjug",
        ("B", "ig"): "Big",
    }
    tokens = tokenizer.tokenize(text=text, vocabulary=vocab_token_to_index, merge_list=merge_list)
    expected_tokens = [
        263,
        262,
        44,
        260,
        259,
        46,
    ]


def _byte_vocab() -> Dict[int, str]:
    return dict(text_processing.BYTE_TO_UNICODE_REPRESENTATION)
