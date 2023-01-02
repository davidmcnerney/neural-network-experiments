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

    actual_vocab, actual_merges = builder.build_vocabulary_and_merge_list(
        training_text=training_text,
        count_merges=1000,
    )

    print("\n\n")
    _summarize_vocab(_base_bytes_removed(actual_vocab))
    _summarize_merges(actual_merges)
    assert False


def _byte_vocab() -> Dict[int, str]:
    return dict(text_processing.BYTE_TO_UNICODE_REPRESENTATION)


def _base_bytes_removed(vocab: builder.Vocabulary) -> builder.Vocabulary:
    copy = dict(vocab)
    for index in range(256):
        del copy[index]
    return copy


def _summarize_vocab(vocab: builder.Vocabulary) -> None:
    for index, string in vocab.items():
        print(f"{index:6}: {string}")


def _summarize_merges(merges: List[builder.Merge]) -> None:
    for first, second, merged in merges:
        print(f"{first} + {second} -> {merged}")
