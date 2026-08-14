"""分层召回组包：L3 sticky → L2 相关 → L1 检索填满。"""

from __future__ import annotations

from ..memory_layers import (
    LAYER_L2,
    LAYER_L3,
    effective_layer,
    sticky_l3_atoms,
)
from ..models import Atom
from .base import RecallHit, RecallOutcome
from .bm25 import Bm25Recall
from .tokenize import tokenize

_L3_MAX = 2
_L2_MAX = 2


def _dedupe(hits: list[RecallHit]) -> list[RecallHit]:
    seen: set[str] = set()
    out: list[RecallHit] = []
    for h in hits:
        if h.atom.key in seen:
            continue
        seen.add(h.atom.key)
        out.append(h)
    return out


def _rank_l2(
    atoms_l2: list[Atom], query: str, *, limit: int
) -> list[RecallHit]:
    if not atoms_l2 or limit <= 0:
        return []
    # 复用 BM25，仅在 L2 子集上打分
    outcome = Bm25Recall().retrieve(atoms_l2, query, max(limit, 5))
    if outcome.hits:
        return outcome.hits[:limit]
    # 无命中时按更新时间兜底
    ordered = sorted(
        atoms_l2, key=lambda a: a.updated_at, reverse=True
    )
    return [RecallHit(atom=a, score=None) for a in ordered[:limit]]


def compose_layered(
    atoms: list[Atom],
    query: str,
    *,
    l1_hits: list[RecallHit],
    max_atoms: int,
    derived_ids: set[int] | None = None,
) -> list[RecallHit]:
    """L3 sticky（≤2）+ L2 相关（≤2）+ L1 检索，去重后截断。"""
    if max_atoms <= 0:
        return []

    l3 = sticky_l3_atoms(atoms, limit=_L3_MAX)
    l3_hits = [RecallHit(atom=a, score=None) for a in l3]

    atoms_l2 = [
        a
        for a in atoms
        if effective_layer(a, derived_ids=derived_ids) == LAYER_L2
        and a.key not in {h.atom.key for h in l3_hits}
    ]
    # 疑问词很少时仍给 L2 名额
    l2_budget = min(_L2_MAX, max(0, max_atoms - len(l3_hits)))
    if tokenize(query):
        l2_hits = _rank_l2(atoms_l2, query, limit=l2_budget)
    else:
        l2_hits = [
            RecallHit(atom=a, score=None)
            for a in sorted(
                atoms_l2, key=lambda x: x.updated_at, reverse=True
            )[:l2_budget]
        ]

    have = {h.atom.key for h in l3_hits} | {h.atom.key for h in l2_hits}
    l1_extra = [h for h in l1_hits if h.atom.key not in have]
    # 若 L1 命中其实是 L2/L3，仍可进填空（已在 have 则跳过）
    merged = _dedupe(l3_hits + l2_hits + l1_extra)
    return merged[:max_atoms]


def partition_by_layer(
    atoms: list[Atom],
    *,
    derived_ids: set[int] | None = None,
) -> tuple[list[Atom], list[Atom], list[Atom]]:
    """返回 (l1, l2, l3)。"""
    l1: list[Atom] = []
    l2: list[Atom] = []
    l3: list[Atom] = []
    for a in atoms:
        layer = effective_layer(a, derived_ids=derived_ids)
        if layer == LAYER_L3:
            l3.append(a)
        elif layer == LAYER_L2:
            l2.append(a)
        else:
            l1.append(a)
    return l1, l2, l3


def layered_retrieve(
    atoms: list[Atom],
    query: str,
    *,
    strategy_outcome: RecallOutcome,
    max_atoms: int,
    derived_ids: set[int] | None = None,
) -> list[RecallHit]:
    """在已有 L1 策略召回结果上组分层包。"""
    return compose_layered(
        atoms,
        query,
        l1_hits=list(strategy_outcome.hits),
        max_atoms=max_atoms,
        derived_ids=derived_ids,
    )
