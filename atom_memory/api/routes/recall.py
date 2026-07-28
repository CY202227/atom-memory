"""召回路由：statement 默认注入 + 字符预算 + 可选近期 pending source。"""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ...db import get_session
from ...llm import ChatLLM
from ...models import Space
from ...recall import (
    clip_by_budget,
    fallback_hits,
    is_inventory_query,
    merge_context,
    render_detail_block,
    render_recent_sources,
    render_statement_block,
)
from ...recall.base import RecallHit
from ...recall.llm import RecallError
from ...repositories import atom_repo, source_repo
from .. import schemas
from ..deps import build_recall_strategy, get_llm, get_space, require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


def _hit_cost(h: RecallHit, detail: str) -> int:
    if detail == "full":
        return len(h.atom.statement) + len(h.atom.detail or "")
    return len(h.atom.statement)


@router.post("/spaces/{space_uid}/recall", response_model=schemas.RecallResponse)
def recall(
    payload: schemas.RecallRequest,
    space: Space = Depends(get_space),
    session: Session = Depends(get_session),
    llm: ChatLLM = Depends(get_llm),
):
    atoms = atom_repo.list_active(session, space.id)
    strategy = build_recall_strategy(payload.method, llm)
    try:
        outcome = strategy.retrieve(atoms, payload.query, payload.max_atoms)
    except RecallError as e:
        raise HTTPException(status_code=502, detail=f"recall LLM failed: {e}")

    hits_raw = list(outcome.hits)
    need_fallback = (not hits_raw or is_inventory_query(payload.query)) and atoms
    if need_fallback:
        exclude = {h.atom.key for h in hits_raw}
        fill = fallback_hits(
            atoms,
            exclude_keys=exclude,
            limit=max(0, payload.max_atoms - len(hits_raw)),
        )
        if is_inventory_query(payload.query) and not hits_raw:
            hits_raw = fill
        else:
            hits_raw.extend(fill)
        hits_raw = hits_raw[: payload.max_atoms]

    pre_budget = list(hits_raw)
    hits = clip_by_budget(hits_raw, payload.budget_chars)

    recent_lines: list[str] = []
    if payload.include_recent_sources:
        recent = source_repo.list_recent_for_recall(session, space.id)
        recent_lines = render_recent_sources(recent)

    if payload.detail == "full":
        kept = []
        used = 0
        for h in hits:
            cost = _hit_cost(h, "full")
            if kept and used + cost > payload.budget_chars:
                break
            kept.append(h)
            used += cost
        hits = kept
        chars_used = used
        block = render_detail_block([h.atom for h in hits])
        hit_out = [
            schemas.RecallHitOut(
                key=h.atom.key,
                kind=h.atom.kind,
                statement=h.atom.statement,
                score=h.score,
                happened_on=h.atom.happened_on,
                detail=h.atom.detail,
            )
            for h in hits
        ]
    else:
        chars_used = sum(_hit_cost(h, "statement") for h in hits)
        block = render_statement_block(hits)
        hit_out = [
            schemas.RecallHitOut(
                key=h.atom.key,
                kind=h.atom.kind,
                statement=h.atom.statement,
                score=h.score,
                happened_on=h.atom.happened_on,
            )
            for h in hits
        ]

    atoms_clipped = max(0, len(pre_budget) - len(hits))

    return schemas.RecallResponse(
        method=payload.method,
        hits=hit_out,
        context_block=merge_context(block, recent_lines),
        chars_used=chars_used,
        atoms_clipped=atoms_clipped,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
    )
