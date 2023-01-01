from typing import Dict, List, Tuple


Vocab = Dict[int, str]
MergeList =  List[Tuple[str, str, str]]


def build_vocab(training_text: str) -> Tuple[Vocab, MergeList]:
    return {}, []
