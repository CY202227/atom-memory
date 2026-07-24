"""BM25 召回：文档 = statement×3 + detail。"""

import math
from collections import Counter

from ..models import Atom
from .base import RecallHit, RecallOutcome
from .tokenize import tokenize

_K1 = 1.5
_B = 0.75


def _doc_tokens(atom: Atom) -> list[str]:
    return tokenize(atom.statement) * 3 + tokenize(atom.detail or "")


class Bm25Recall:
    def retrieve(self, atoms: list[Atom], query: str, max_atoms: int) -> RecallOutcome:
        terms = tokenize(query)
        if not terms or not atoms:
            return RecallOutcome()

        docs = [Counter(_doc_tokens(a)) for a in atoms]
        doc_lens = [sum(c.values()) for c in docs]
        avg_len = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0
        n = len(docs)

        df = Counter()
        for term in set(terms):
            df[term] = sum(1 for d in docs if term in d)

        scored: list[RecallHit] = []
        for atom, doc, dl in zip(atoms, docs, doc_lens):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf * (_K1 + 1) / (
                    tf + _K1 * (1 - _B + _B * dl / avg_len)
                )
            if score > 0:
                scored.append(RecallHit(atom=atom, score=round(score, 4)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return RecallOutcome(hits=scored[:max_atoms])
