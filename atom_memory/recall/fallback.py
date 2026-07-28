"""泛查询 / 空召回保底：最近更新的高价值 kind。"""

from __future__ import annotations

import re

from ..models import Atom, AtomKind
from .base import RecallHit

_INVENTORY_QUERY_RE = re.compile(
    r"(你记得什么|记得什么|记得哪些|记得我什么|我的情况|你都记得|"
    r"what do you remember|what('s| is) on (your|my) mind|"
    r"tell me what you know about me)",
    re.IGNORECASE,
)

_FALLBACK_KINDS = (
    AtomKind.person,
    AtomKind.event,
    AtomKind.lesson,
    AtomKind.self,
)


def is_inventory_query(query: str) -> bool:
    """清单式问句：用户想听「你还记得哪些」，而非关键词检索。"""
    return bool(_INVENTORY_QUERY_RE.search(query or ""))


def fallback_hits(
    atoms: list[Atom],
    *,
    exclude_keys: set[str] | None = None,
    limit: int = 5,
) -> list[RecallHit]:
    """按 updated_at 倒序，优先 person/event/lesson/self，填满至 limit。"""
    if limit <= 0 or not atoms:
        return []
    exclude = exclude_keys or set()
    preferred = [
        a
        for a in atoms
        if a.key not in exclude and a.kind in _FALLBACK_KINDS
    ]
    preferred.sort(key=lambda a: a.updated_at, reverse=True)
    out: list[RecallHit] = [
        RecallHit(atom=a, score=None) for a in preferred[:limit]
    ]
    if len(out) >= limit:
        return out
    have = {h.atom.key for h in out} | exclude
    rest = [a for a in atoms if a.key not in have]
    rest.sort(key=lambda a: a.updated_at, reverse=True)
    for a in rest:
        if len(out) >= limit:
            break
        out.append(RecallHit(atom=a, score=None))
    return out
