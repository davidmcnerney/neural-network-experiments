import json

from independent.gpt.bpe import helpers
from independent.gpt.bpe import type_definitions


def invert_vocabulary(vocabulary_by_index: type_definitions.VocabularyByIndex) -> type_definitions.VocabularyByToken:
    return helpers.invert_dictionary(vocabulary_by_index)


def remove_base_byte_vocab(vocab: type_definitions.VocabularyByIndex) -> type_definitions.VocabularyByIndex:
    copy = dict(vocab)
    for index in range(256):
        del copy[index]
    return copy


def summarize_vocabulary(vocab: type_definitions.VocabularyByIndex) -> None:
    for index, string in vocab.items():
        print(f"{index:6}: {string}")


def summarize_merges(merges: type_definitions.MergeList) -> None:
    for t, merged in merges.items():
        first, second = t
        print(f"{first} + {second} -> {merged}")


def serialize_vocabulary(vocab: type_definitions.VocabularyByIndex) -> str:
    return json.dumps(vocab, indent=3)


def serialize_merges(merges: type_definitions.MergeList) -> str:
    output_string = ""
    for t, _ in merges.items():
        first, second = t
        output_string += f"{first} {second}\n"
    return output_string


def deserialize_vocabulary(string: str) -> type_definitions.VocabularyByIndex:
    return json.loads(string)


def deserialize_merge_list(string: str) -> type_definitions.MergeList:
    merge_list: type_definitions.MergeList = {}
    for line in string.splitlines():
        first, second = line.split(" ")
        merged = first + second
        merge_list[first, second] = merged
    return merge_list


def load_vocabulary_from_file(filename: str) -> type_definitions.VocabularyByIndex:
    with open(filename) as file:
        contents = file.read()
        return deserialize_vocabulary(contents)


def load_merge_list_from_file(filename: str) -> type_definitions.MergeList:
    with open(filename) as file:
        contents = file.read()
        return deserialize_merge_list(contents)
