"""召回结果渲染与预算裁剪。"""

from ..models import Atom, Source
from .base import RecallHit

_EVIDENCE_EXCERPT_MAX = 200
_EXPAND_BUDGET_DEFAULT = 1200


def render_statement_block(hits: list[RecallHit]) -> str:
    if not hits:
        return ""
    parts = ["<recalled_memory>"]
    for h in hits:
        a = h.atom
        when = f" {a.happened_on:%Y-%m-%d}" if a.happened_on else ""
        parts.append(f'<item key="{a.key}" kind="{a.kind.value}">{when} {a.statement}</item>')
    parts.append("</recalled_memory>")
    return "\n".join(parts)


def render_detail_block(atoms: list[Atom]) -> str:
    if not atoms:
        return ""
    parts = ["<recalled_memory>"]
    for a in atoms:
        parts.append(
            f'<memory key="{a.key}" kind="{a.kind.value}" '
            f'statement="{a.statement}">\n{a.detail or a.statement}\n</memory>'
        )
    parts.append("</recalled_memory>")
    return "\n".join(parts)


def clip_by_budget(hits: list[RecallHit], budget_chars: int) -> list[RecallHit]:
    """按 statement 字符预算裁剪，至少保留第一条（若有）。"""
    if budget_chars <= 0 or not hits:
        return []
    out: list[RecallHit] = []
    used = 0
    for h in hits:
        cost = len(h.atom.statement)
        if out and used + cost > budget_chars:
            break
        out.append(h)
        used += cost
    return out


def truncate_source(content: str, limit: int = _EVIDENCE_EXCERPT_MAX) -> str:
    content = content.strip()
    if len(content) <= limit:
        return content
    return content[: limit - 1] + "…"


def render_recent_sources(sources: list[Source]) -> list[str]:
    """pending/高显著临时行（不进 hits，拼进 context 或单独字段）。"""
    lines = []
    for s in sources:
        excerpt = truncate_source(s.content, 80)
        lines.append(f'- [pending source_id={s.id} kind={s.kind.value}] {excerpt}')
    return lines


def merge_context(statement_block: str, recent_lines: list[str]) -> str:
    if not recent_lines:
        return statement_block
    recent = "<recent_sources>\n" + "\n".join(recent_lines) + "\n</recent_sources>"
    if not statement_block:
        return recent
    return statement_block + "\n" + recent
