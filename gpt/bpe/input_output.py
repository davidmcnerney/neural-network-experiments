import json
from pathlib import Path
from typing import Union

from gpt.bpe import helpers
from gpt.bpe import type_definitions


def invert_vocabulary(vocabulary_by_index: type_definitions.VocabularyByIndex) -> type_definitions.VocabularyByToken:
    return helpers.invert_dictionary(vocabulary_by_index)


def remove_base_byte_vocab(vocab: type_definitions.VocabularyByToken) -> type_definitions.VocabularyByToken:
    copy = dict(vocab)
    for token, index in vocab.items():
        if index < 256:
            del copy[token]
    return copy


def summarize_vocabulary(vocab: type_definitions.VocabularyByToken) -> None:
    for token, index in vocab.items():
        print(f"{token}: {index:6}")


def summarize_merges(merges: type_definitions.MergeList) -> None:
    for t, merged in merges.items():
        first, second = t
        print(f"{first} + {second} -> {merged}")


def serialize_vocabulary(vocab: type_definitions.VocabularyByToken) -> str:
    return json.dumps(vocab, indent=3)


def serialize_merge_list(merges: type_definitions.MergeList) -> str:
    output_string = ""
    for t, _ in merges.items():
        first, second = t
        output_string += f"{first} {second}\n"
    return output_string


def deserialize_vocabulary(string: str) -> type_definitions.VocabularyByToken:
    return json.loads(string)


def deserialize_merge_list(string: str) -> type_definitions.MergeList:
    merge_list: type_definitions.MergeList = {}
    for line in string.splitlines():
        first, second = line.split(" ")
        merged = first + second
        merge_list[first, second] = merged
    return merge_list


def load_vocabulary_from_file(filename: Union[str,Path]) -> type_definitions.VocabularyByToken:
    with open(filename) as file:
        contents = file.read()
        return deserialize_vocabulary(contents)


def load_merge_list_from_file(filename: Union[str,Path]) -> type_definitions.MergeList:
    with open(filename) as file:
        contents = file.read()
        return deserialize_merge_list(contents)
