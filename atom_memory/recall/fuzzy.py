"""模糊词频召回。"""

from collections import Counter

from ..models import Atom
from .base import RecallHit, RecallOutcome
from .tokenize import tokenize


def _doc_tokens(atom: Atom) -> list[str]:
    return tokenize(atom.statement) * 3 + tokenize(atom.detail or "")


class FuzzyRecall:
    def retrieve(self, atoms: list[Atom], query: str, max_atoms: int) -> RecallOutcome:
        terms = tokenize(query)
        if not terms or not atoms:
            return RecallOutcome()
        q = Counter(terms)
        scored: list[RecallHit] = []
        for atom in atoms:
            doc = Counter(_doc_tokens(atom))
            score = sum(doc[t] * w for t, w in q.items() if t in doc)
            if score > 0:
                scored.append(RecallHit(atom=atom, score=float(score)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return RecallOutcome(hits=scored[:max_atoms])
