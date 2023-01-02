from typing import Dict, Tuple


VocabularyByIndex = Dict[int, str]          # token index integer -> token's Unicode byte representation
VocabularyByToken = Dict[str, int]          # token's Unicode byte representation -> token index integer
MergeList = Dict[Tuple[str, str], str]      # two token Unicode byte representations -> their merged token
