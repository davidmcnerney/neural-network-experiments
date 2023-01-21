from textwrap import dedent
from typing import Dict

from independent.gpt.bpe import builder
from independent.gpt.bpe import input_output
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


def test_to_unicode_byte_representation():
    string = "Big jug, dig dug."
    expected = "BigĠjug,ĠdigĠdug."
    assert text_processing.to_unicode_byte_representation(string=string) == expected


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

    expected_vocab = dedent("""
        {
           "\\u0120d": 256,
           "ug": 257,
           "ig": 258,
           "\\u0120dug": 259,
           "\\u0120dig": 260,
           "jug": 261,
           "\\u0120jug": 262,
           "Big": 263
        }
    """)
    actual_vocab = input_output.serialize_vocabulary(input_output.remove_base_byte_vocab(vocab))
    assert "\n" + actual_vocab + "\n" == expected_vocab

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
    actual_merges = input_output.serialize_merge_list(merges)
    assert "\n" + actual_merges == expected_merges


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
    vocab_token_to_index = input_output.invert_vocabulary(vocab_index_to_token)
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
    assert tokens == expected_tokens


def test_detokenize():
    tokens = [
        263,
        262,
        44,
        260,
        259,
        46,
    ]
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
    text = tokenizer.detokenize(tokens, vocab_index_to_token)
    expected_text = """Big jug, dig dug."""
    assert text == expected_text


def _byte_vocab() -> Dict[int, str]:
    return dict(text_processing.BYTE_TO_UNICODE_REPRESENTATION)
