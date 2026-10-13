"""固化写路径的 canonical key 与同类型精确去重。"""

from __future__ import annotations

import re
from typing import Any

from ..linking import keyify
from ..models import Atom, AtomKind

# 称呼：一律落到此 key
CANONICAL_PREFERRED_NAME = "user-preferred-name"
_PREFERRED_NAME_ALIASES = frozenset(
    {
        CANONICAL_PREFERRED_NAME,
        "user-preference-name",
        "preferred-name",
        "user-name",
        "call-me",
        "call-me-name",
        "nickname",
        "称呼",
        "称呼偏好",
    }
)
_PREFERRED_NAME_KEY_RE = re.compile(
    r"(preferred.?name|preference.?name|call.?me|nickname|称呼)",
    re.IGNORECASE,
)

# 核心人设：新建时规范到 persona；已有近义 self 则并入
CANONICAL_PERSONA = "persona"
_PERSONA_ALIASES = frozenset(
    {
        CANONICAL_PERSONA,
        "identity-definition",
        "identity",
        "self-persona",
        "persona-catgirl",
        "catgirl",
        "人设",
        "口癖",
    }
)
_PERSONA_KEY_RE = re.compile(
    r"^(persona|identity|catgirl|人设|口癖)",
    re.IGNORECASE,
)

def _norm_text(text: str) -> str:
    return " ".join((text or "").split())


def resolve_write_key(
    *,
    key: str,
    kind: AtomKind | str,
    statement: str,
    active_atoms: list[Atom],
    detail: str = "",
    happened_on=None,
) -> tuple[str, str | None]:
    """将拟写入 key 规范到 canonical / 同类型精确重复的已有 key。

    Returns:
        (final_key, merge_note)；无改写时 merge_note 为 None。
    """
    raw = keyify(key or "")
    if not raw:
        return raw, None

    kind_val = kind.value if isinstance(kind, AtomKind) else str(kind)
    by_key = {a.key: a for a in active_atoms}

    # --- 称呼（仅别名/key 形态；不按 statement 误伤人物页如 lao-zhang）---
    if raw in _PREFERRED_NAME_ALIASES or (raw.startswith("user-") and _PREFERRED_NAME_KEY_RE.search(raw)):
        if raw != CANONICAL_PREFERRED_NAME:
            return CANONICAL_PREFERRED_NAME, f"merged_into:{CANONICAL_PREFERRED_NAME}"
        return CANONICAL_PREFERRED_NAME, None

    # --- 人设 / self 身份 ---
    if kind_val == AtomKind.self.value and (
        raw in _PERSONA_ALIASES or _PERSONA_KEY_RE.search(raw)
    ):
        existing = _find_existing_persona(active_atoms)
        if existing is not None:
            note = f"merged_into:{existing}" if existing != raw else None
            return existing, note
        if raw != CANONICAL_PERSONA:
            return CANONICAL_PERSONA, f"merged_into:{CANONICAL_PERSONA}"
        return CANONICAL_PERSONA, None

    # Text similarity cannot prove semantic identity. Reuse only exact same-kind facts.
    dup = _find_exact_duplicate(statement, active_atoms, prefer_kind=kind_val, detail=detail, happened_on=happened_on)
    if dup is not None and dup != raw and raw not in by_key:
        return dup, f"merged_into:{dup}"

    return raw, None


def _find_existing_persona(active_atoms: list[Atom]) -> str | None:
    for a in active_atoms:
        if a.kind != AtomKind.self:
            continue
        if a.key in _PERSONA_ALIASES or _PERSONA_KEY_RE.search(a.key):
            return a.key
    return None


def _find_exact_duplicate(
    statement: str,
    active_atoms: list[Atom],
    *,
    prefer_kind: str | None = None,
    detail: str = "",
    happened_on=None,
) -> str | None:
    needle = _norm_text(statement)
    if len(needle) < 6:
        return None
    for a in active_atoms:
        existing_date = a.happened_on.isoformat() if a.happened_on else None
        requested_date = str(happened_on) if happened_on else None
        if a.kind.value == prefer_kind and needle == _norm_text(a.statement) and _norm_text(detail) == _norm_text(a.detail) and existing_date == requested_date:
            return a.key
    return None


def apply_canonical_to_op(
    op: dict[str, Any], active_atoms: list[Atom]
) -> tuple[dict[str, Any], str | None]:
    """规范 op['key']；merge 时写入 change_reason。"""
    if op.get("op") != "upsert":
        return op, None
    kind = op.get("kind") or AtomKind.belief.value
    statement = op.get("statement") or ""
    key = op.get("key") or ""
    final, note = resolve_write_key(
        key=key,
        kind=kind,
        statement=statement,
        active_atoms=active_atoms,
        detail=op.get("detail") or "",
        happened_on=op.get("happened_on"),
    )
    if not final:
        return op, None
    op = dict(op)
    op["key"] = final
    if note:
        reason = (op.get("change_reason") or "").strip()
        op["change_reason"] = f"{reason}; {note}".strip("; ").strip()
    # 称呼强制 person；并入已有 persona 时沿用其 kind
    if final == CANONICAL_PREFERRED_NAME:
        op["kind"] = AtomKind.person.value
    elif note and note.startswith("merged_into:"):
        target = note.split(":", 1)[1]
        existing = next((a for a in active_atoms if a.key == target), None)
        if existing is not None:
            op["kind"] = existing.kind.value
        elif target == CANONICAL_PERSONA:
            op["kind"] = AtomKind.self.value
    return op, note
