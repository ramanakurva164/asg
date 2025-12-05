# app/utils.py
import re
from typing import List, Union


def simple_tokenize_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def best_sentences_for_query(
    texts: Union[str, List[str]],
    query: Union[str, List[str]],
    k: int = 2
) -> List[str]:
    """
    Robust version:
    - texts can be a single string OR a list of strings (multiple docs)
    - query can be a string OR a list (we join it)
    - never crashes with '... has no attribute lower'
    """

    # 🔹 Normalize texts → single big string
    if isinstance(texts, list):
        texts = "\n".join([t for t in texts if isinstance(t, str)])
    elif not isinstance(texts, str):
        texts = str(texts)

    # 🔹 Normalize query → single string
    if isinstance(query, list):
        query = " ".join([str(q) for q in query])
    elif not isinstance(query, str):
        query = str(query)

    # now both are safe strings
    sents = simple_tokenize_sentences(texts)

    qtok = set(re.findall(r"\w+", query.lower()))
    scored = []
    for s in sents:
        stok = set(re.findall(r"\w+", s.lower()))
        scored.append((len(qtok & stok), s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:k]] if scored else sents[:k]
