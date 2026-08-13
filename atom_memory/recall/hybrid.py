"""Hybrid RRF：BM25 + embedding 过取后按倒数排名融合。"""

from __future__ import annotations

from sqlmodel import Session

from ..models import Atom
from .base import RecallHit, RecallOutcome
from .bm25 import Bm25Recall
from .embedding import EmbeddingRecall, Embedder


_RRF_K = 60


def _overfetch(max_atoms: int) -> int:
    return max(max_atoms * 4, 20)


def rrf_fuse(
    ranked_lists: list[list[RecallHit]],
    *,
    k: int = _RRF_K,
    limit: int,
) -> list[RecallHit]:
    """按 atom.key 融合；同分保留更高单路分作展示。"""
    scores: dict[str, float] = {}
    best: dict[str, RecallHit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            key = hit.atom.key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            prev = best.get(key)
            if prev is None or (hit.score or 0.0) > (prev.score or 0.0):
                best[key] = hit
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: list[RecallHit] = []
    for key, fused in ordered[:limit]:
        hit = best[key]
        out.append(RecallHit(atom=hit.atom, score=round(fused, 6)))
    return out


class HybridRRF:
    def __init__(
        self,
        session: Session,
        embedder: Embedder,
        *,
        embed_detail_chars: int = 200,
        rrf_k: int = _RRF_K,
    ):
        self._bm25 = Bm25Recall()
        self._embedding = EmbeddingRecall(
            session, embedder, embed_detail_chars=embed_detail_chars
        )
        self._rrf_k = rrf_k

    def retrieve(
        self, atoms: list[Atom], query: str, max_atoms: int
    ) -> RecallOutcome:
        if not atoms or not query.strip():
            return RecallOutcome()
        n = _overfetch(max_atoms)
        bm25 = self._bm25.retrieve(atoms, query, n)
        emb = self._embedding.retrieve(atoms, query, n)
        fused = rrf_fuse(
            [bm25.hits, emb.hits], k=self._rrf_k, limit=max_atoms
        )
        return RecallOutcome(hits=fused)
