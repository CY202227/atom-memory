"""向量召回：statement + detail 截断 → 惰性 embed → 余弦排序。"""

from __future__ import annotations

import math
from typing import Protocol

from sqlmodel import Session

from ..models import Atom
from ..repositories import embedding_repo
from .base import RecallHit, RecallOutcome


class Embedder(Protocol):
    model_tag: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def atom_embed_text(atom: Atom, detail_chars: int) -> str:
    detail = (atom.detail or "")[:detail_chars]
    return f"{atom.statement}\n{detail}".rstrip()


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (na * nb)


class EmbeddingRecall:
    """需要可写 session：缺缓存时 batch embed 并 upsert。"""

    def __init__(
        self,
        session: Session,
        embedder: Embedder,
        *,
        embed_detail_chars: int = 200,
    ):
        self._session = session
        self._embedder = embedder
        self._detail_chars = embed_detail_chars

    def retrieve(
        self, atoms: list[Atom], query: str, max_atoms: int
    ) -> RecallOutcome:
        if not atoms or not query.strip():
            return RecallOutcome()

        vectors = self._ensure_vectors(atoms)
        q_vecs = self._embedder.embed([query.strip()])
        if not q_vecs:
            return RecallOutcome()
        q = q_vecs[0]

        scored: list[RecallHit] = []
        for atom in atoms:
            if atom.id is None:
                continue
            vec = vectors.get(atom.id)
            if vec is None:
                continue
            cos = cosine(q, vec)
            score = (cos + 1.0) / 2.0
            scored.append(RecallHit(atom=atom, score=round(score, 6)))
        scored.sort(key=lambda h: h.score or 0.0, reverse=True)
        return RecallOutcome(hits=scored[:max_atoms])

    def _ensure_vectors(self, atoms: list[Atom]) -> dict[int, list[float]]:
        ids = [a.id for a in atoms if a.id is not None]
        cached = embedding_repo.get_for_atoms(self._session, ids)
        tag = self._embedder.model_tag
        out: dict[int, list[float]] = {}
        missing: list[Atom] = []

        for atom in atoms:
            if atom.id is None:
                continue
            row = cached.get(atom.id)
            if (
                row is None
                or row.model_tag != tag
                or not row.vector
                or row.dim != len(row.vector)
            ):
                missing.append(atom)
            else:
                out[atom.id] = list(row.vector)

        if not missing:
            return out

        texts = [atom_embed_text(a, self._detail_chars) for a in missing]
        new_vecs = self._embedder.embed(texts)
        for atom, vec in zip(missing, new_vecs):
            if atom.id is None:
                continue
            embedding_repo.upsert(
                self._session,
                atom_id=atom.id,
                space_id=atom.space_id,
                model_tag=tag,
                vector=vec,
                existing=cached.get(atom.id),
            )
            out[atom.id] = vec
        self._session.flush()
        self._session.commit()
        return out
