"""Bounded retrieval of retained source excerpts, separate from current atom assertions."""
from __future__ import annotations

from ..models import Atom, AtomKind, Source
from .bm25 import Bm25Recall
from ..temporal import source_time_anchor, derive_time_facts


def search_source_excerpts(sources: list[Source], query: str, max_sources: int, budget_chars: int) -> list[dict]:
    candidates, lookup = [], {}
    for source in sources:
        lines = [line.strip() for line in source.content.splitlines() if line.strip()]
        for index, line in enumerate(lines):
            if line.startswith("[session_time]"):
                continue
            # Long paragraphs are bounded independently rather than silently truncated from the index.
            for offset in range(0, len(line), 600):
                segment = line[offset:offset + 600]
                key = f"source-{source.id}-{index}-{offset}"
                candidates.append(Atom(space_id=source.space_id, key=key, kind=AtomKind.event,
                    statement=segment[:80], detail=segment))
                lookup[key] = (source, lines, index, segment)
    ranked = Bm25Recall().retrieve(candidates, query, max(max_sources * 8, 32)).hits
    results, seen, used = [], set(), 0
    for hit in ranked:
        source, lines, index, segment = lookup[hit.atom.key]
        if source.id in seen:
            continue
        anchor = next((line for line in reversed(lines[:index + 1]) if line.startswith("[session_time]")), "")
        # Include the matching passage first; preserve the source's time anchor and one preceding line.
        preceding = lines[index - 1] if index > 0 and not lines[index - 1].startswith("[session_time]") else ""
        excerpt = "\n".join(value for value in (anchor, segment, preceding) if value)[:800]
        time_anchor = source_time_anchor(anchor)
        time_facts = derive_time_facts(None, segment, source_anchor=time_anchor) if time_anchor else []
        if time_facts:
            excerpt = (excerpt[:600] + "\n[derived_time] " + "; ".join(time_facts))[:800]
        remaining = budget_chars - used
        if remaining <= 0:
            break
        excerpt = excerpt[:remaining]
        if not excerpt:
            break
        results.append({"source_id": source.id, "kind": source.kind.value, "status": source.status.value,
            "occurred_at": source.occurred_at.isoformat(), "excerpt": excerpt, "score": hit.score})
        seen.add(source.id)
        used += len(excerpt)
        if len(results) >= max_sources:
            break
    return results
