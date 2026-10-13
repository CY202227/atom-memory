"""Resolve source time from current evidence, rather than processing time or older revisions."""
from ..repositories.evidence_repo import current_sources
from ..temporal import source_time_anchor
import re


def evidence_time_anchor(session, atom):
    original = (atom.statement + "\n" + (atom.detail or "")).split(" | time_facts=", 1)[0]
    if not re.search(r"\b(?:yesterday|today|tomorrow|last\s+(?:year|week|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|next\s+month|years?)\b|去年|昨天|今天|明天|上周|上个周|下个月|年前|年了", original, re.IGNORECASE):
        return None
    sources = current_sources(session, atom)
    if not sources:
        return None
    content = "\n".join(source.content for source in sources)
    if "[session_time]" in content:
        return source_time_anchor(content)
    dates = {source.occurred_at.date() for source in sources}
    return next(iter(dates)) if len(dates) == 1 else None
