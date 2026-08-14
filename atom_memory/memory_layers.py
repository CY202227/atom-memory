"""Chat Memory 分层：L0=source，L1/L2/L3 在 atom.memory_layer。

设计参考腾讯 TencentDB Agent Memory 的 L0–L3 叙事；实现独立。
"""

from __future__ import annotations

from .consolidation.canonical import CANONICAL_PERSONA, CANONICAL_PREFERRED_NAME
from .models import Atom, AtomKind

LAYER_L1 = 1
LAYER_L2 = 2
LAYER_L3 = 3

_L3_KEYS = frozenset({CANONICAL_PERSONA, CANONICAL_PREFERRED_NAME})


def is_l3_key(key: str) -> bool:
    return (key or "") in _L3_KEYS


def infer_layer(
    atom: Atom,
    *,
    derived_ids: set[int] | None = None,
) -> int:
    """推断 memory_layer（用于回填 / 读时兜底）。

    优先级：显式字段（若已是 1–3 且非默认歧义则仍可被规则纠正）→
    L3 key/kind → L2 derived_from → L1。
    """
    if is_l3_key(atom.key) or atom.kind == AtomKind.self:
        return LAYER_L3
    if derived_ids is not None and atom.id is not None and atom.id in derived_ids:
        return LAYER_L2
    # 已显式标为 2/3 时尊重（synthesize / persona 写入）
    if atom.memory_layer in (LAYER_L2, LAYER_L3):
        return int(atom.memory_layer)
    return LAYER_L1


def default_layer_for_write(
    *,
    key: str,
    kind: AtomKind | str,
    explicit: int | None = None,
) -> int:
    """写路径默认层：consolidate→1（persona key→3）；可显式覆盖。"""
    if explicit in (LAYER_L1, LAYER_L2, LAYER_L3):
        return int(explicit)
    kind_val = kind.value if isinstance(kind, AtomKind) else str(kind)
    if is_l3_key(key) or kind_val == AtomKind.self.value:
        return LAYER_L3
    return LAYER_L1


def effective_layer(
    atom: Atom,
    *,
    derived_ids: set[int] | None = None,
) -> int:
    """召回用：优先已写字段，缺省或 0 时按规则推断。"""
    stored = int(atom.memory_layer or 0)
    if stored in (LAYER_L1, LAYER_L2, LAYER_L3):
        # 旧库默认全是 1：若其实是 L3 key / derived，纠正
        if stored == LAYER_L1:
            inferred = infer_layer(atom, derived_ids=derived_ids)
            if inferred != LAYER_L1:
                return inferred
        return stored
    return infer_layer(atom, derived_ids=derived_ids)


def sticky_l3_atoms(atoms: list[Atom], *, limit: int = 2) -> list[Atom]:
    """L3 sticky：称呼优先，再 persona / self。"""
    by_key = {a.key: a for a in atoms}
    out: list[Atom] = []
    for key in (CANONICAL_PREFERRED_NAME, CANONICAL_PERSONA):
        atom = by_key.get(key)
        if atom is not None:
            out.append(atom)
        if len(out) >= limit:
            return out[:limit]
    for atom in atoms:
        if atom.key in {a.key for a in out}:
            continue
        if atom.kind == AtomKind.self or effective_layer(atom) == LAYER_L3:
            out.append(atom)
        if len(out) >= limit:
            break
    return out[:limit]
