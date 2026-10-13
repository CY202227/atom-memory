"""召回结果渲染与预算裁剪。"""

from datetime import date
from html import escape
import re

from ..models import Atom, Source
from .base import RecallHit
from ..temporal import derive_time_facts

_EVIDENCE_EXCERPT_MAX = 200
_EXPAND_BUDGET_DEFAULT = 1200


def time_context(atom: Atom, *, today: date | None = None, source_anchor: date | None = None) -> str:
    """Expose stored calendar derivations without inventing finer precision."""
    values = []
    if should_prefix_happened_on(atom.happened_on, atom.statement, today=today):
        values.append(atom.happened_on.isoformat())
    marker = " | time_facts="
    suffix = ""
    special = []
    if source_anchor is not None:
        original = (atom.detail or "").split(marker, 1)[0]
        suffix = ", ".join(derive_time_facts(atom.happened_on, atom.statement + "\n" + original, source_anchor=source_anchor, include_contextual_periods=True))
    elif marker in (atom.detail or ""):
        suffix = atom.detail.rsplit(marker, 1)[1]
    if suffix:
        def week(match):
            try:
                anchor = date.fromisoformat(match.group(1))
                special.append(f"week before {anchor.day} {anchor.strftime('%B')} {anchor.year} (week precision)")
            except ValueError:
                pass
            return ""
        suffix = re.sub(r"week before (\d{4}-\d{2}-\d{2})\s*\(last week\)", week, suffix)
        def approximate(match):
            special.append(f"around {match.group(1)} (year precision)")
            return ""
        suffix = re.sub(r"around (\d{4})\s*\(elapsed years; year precision\)", approximate, suffix)
        for value in re.findall(r"(?<!\d)\d{4}(?:-\d{2}(?:-\d{2})?)?(?![\d-])", suffix):
            try:
                parts = value.split("-")
                date(int(parts[0]), int(parts[1]) if len(parts) > 1 else 1, int(parts[2]) if len(parts) > 2 else 1)
            except ValueError:
                continue
            values.append(value)
    values = list(dict.fromkeys(values))
    # A day/month already conveys its containing year; preserve distinct years.
    values = [value for value in values if not any(other.startswith(value + "-") for other in values)]
    precision = {4: "year", 7: "month", 10: "day"}
    return "; ".join([f"{value} ({precision[len(value)]} precision)" for value in values] + list(dict.fromkeys(special)))


def memory_content_cost(atom: Atom, *, detail: str = "statement", source_anchor: date | None = None) -> int:
    return (len(atom.statement) + len(time_context(atom, source_anchor=source_anchor))
        + (10 if source_anchor else 0) + (len(atom.detail or atom.statement) if detail == "full" else 0))


def should_prefix_happened_on(
    happened_on: date | None,
    statement: str,
    *,
    today: date | None = None,
) -> bool:
    """是否在注入前缀 happened_on：跳过写入日噪声与正文已含日期。"""
    if happened_on is None:
        return False
    today = today or date.today()
    if happened_on == today:
        return False
    text = statement or ""
    iso = happened_on.isoformat()
    if iso in text:
        return False
    md = f"{happened_on.month}月{happened_on.day}日"
    md_pad = f"{happened_on.month}月{happened_on.day:02d}日"
    if md in text or md_pad in text:
        return False
    return True


def render_statement_block(
    hits: list[RecallHit],
    *,
    today: date | None = None,
) -> str:
    if not hits:
        return ""
    today = today or date.today()
    parts = ["<recalled_memory>"]
    for h in hits:
        a = h.atom
        temporal = time_context(a, today=today, source_anchor=h.source_anchor)
        when = f"<event_time>{escape(temporal)}</event_time> " if temporal else ""
        source = f"<source_time>{h.source_anchor.isoformat()}</source_time> " if h.source_anchor else ""
        parts.append(
            f'<item key="{escape(a.key, quote=True)}" kind="{a.kind.value}">{source}{when}{escape(a.statement)}</item>'
        )
    parts.append("</recalled_memory>")
    return "\n".join(parts)


def render_detail_block(atoms: list[Atom], *, source_anchors: dict[str, date] | None = None) -> str:
    if not atoms:
        return ""
    parts = ["<recalled_memory>"]
    for a in atoms:
        source_anchor = (source_anchors or {}).get(a.key)
        temporal = time_context(a, source_anchor=source_anchor)
        when = f"<event_time>{escape(temporal)}</event_time>\n" if temporal else ""
        source = f"<source_time>{source_anchor.isoformat()}</source_time>\n" if source_anchor else ""
        parts.append(
            f'<memory key="{escape(a.key, quote=True)}" kind="{a.kind.value}" '
            f'statement="{escape(a.statement, quote=True)}">\n{source}{when}{escape(a.detail or a.statement)}\n</memory>'
        )
    parts.append("</recalled_memory>")
    return "\n".join(parts)


def clip_by_budget(hits: list[RecallHit], budget_chars: int) -> list[RecallHit]:
    """Fit statements and their temporal context within the content budget."""
    if budget_chars <= 0 or not hits:
        return []
    out: list[RecallHit] = []
    used = 0
    for h in hits:
        cost = memory_content_cost(h.atom, source_anchor=h.source_anchor)
        if used + cost > budget_chars:
            continue
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
