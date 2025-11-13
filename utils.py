# app/utils.py
import re
from typing import List

def simple_tokenize_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def best_sentences_for_query(text: str, query: str, k: int = 2) -> List[str]:
    sents = simple_tokenize_sentences(text)
    qtok = set(re.findall(r'\w+', query.lower()))
    scored = []
    for s in sents:
        stok = set(re.findall(r'\w+', s.lower()))
        scored.append((len(qtok & stok), s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for sc,s in scored[:k]] if scored else sents[:k]
