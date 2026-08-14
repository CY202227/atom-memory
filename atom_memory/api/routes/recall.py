"""召回路由：statement 默认注入 + 字符预算 + 可选近期 pending source。"""

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ...db import get_session
from ...embedding import EmbedderError, OpenAICompatEmbedder
from ...llm import ChatLLM
from ...models import AtomLinkKind, Space
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
from ...recall.layered import layered_retrieve, partition_by_layer
from ...recall.llm import RecallError
from ...repositories import atom_link_repo, atom_repo, source_repo
from .. import schemas
from ..deps import (
    build_recall_strategy,
    get_embedder,
    get_llm,
    get_space,
    require_api_key,
)

router = APIRouter(dependencies=[Depends(require_api_key)])

_NEIGHBOR_SCORE_FACTOR = 0.5


def _hit_cost(h: RecallHit, detail: str) -> int:
    if detail == "full":
        return len(h.atom.statement) + len(h.atom.detail or "")
    return len(h.atom.statement)


def _apply_derived_confidence_penalty(
    session: Session, space_id: int, hits: list[RecallHit]
) -> list[RecallHit]:
    """派生 atom 的 score × confidence，同等相关度下事实优先。"""
    derived_ids = atom_link_repo.derived_atom_ids(session, space_id)
    if not derived_ids:
        return hits
    out: list[RecallHit] = []
    for h in hits:
        if h.atom.id in derived_ids and h.score is not None:
            conf = h.atom.confidence if h.atom.confidence is not None else 0.5
            out.append(
                RecallHit(atom=h.atom, score=round(float(h.score) * float(conf), 6))
            )
        else:
            out.append(h)
    out.sort(key=lambda x: x.score or 0.0, reverse=True)
    return out


def _expand_neighbors(
    session: Session,
    hits: list[RecallHit],
    *,
    max_atoms: int,
) -> list[RecallHit]:
    """主命中后沿 about/derived_from 补一跳邻居（score×0.5）。"""
    if not hits or len(hits) >= max_atoms:
        return hits[:max_atoms]
    seed_ids = [h.atom.id for h in hits if h.atom.id is not None]
    peers = atom_link_repo.neighbor_atoms(
        session,
        seed_ids,
        kinds=[AtomLinkKind.about, AtomLinkKind.derived_from],
    )
    have = {h.atom.key for h in hits}
    extras: list[RecallHit] = []
    for atom in peers:
        if atom.key in have:
            continue
        base = min((h.score for h in hits if h.score is not None), default=1.0)
        extras.append(
            RecallHit(
                atom=atom,
                score=round(float(base) * _NEIGHBOR_SCORE_FACTOR, 6),
            )
        )
        have.add(atom.key)
        if len(hits) + len(extras) >= max_atoms:
            break
    return (hits + extras)[:max_atoms]


def _hit_out(h: RecallHit, *, detail: str) -> schemas.RecallHitOut:
    return schemas.RecallHitOut(
        key=h.atom.key,
        kind=h.atom.kind,
        statement=h.atom.statement,
        score=h.score,
        happened_on=h.atom.happened_on,
        detail=h.atom.detail if detail == "full" else None,
        memory_layer=int(h.atom.memory_layer or 1),
    )


@router.post("/spaces/{space_uid}/recall", response_model=schemas.RecallResponse)
def recall(
    payload: schemas.RecallRequest,
    space: Space = Depends(get_space),
    session: Session = Depends(get_session),
    llm: ChatLLM = Depends(get_llm),
    embedder: OpenAICompatEmbedder | None = Depends(get_embedder),
):
    atoms = atom_repo.list_active(session, space.id)
    derived_ids = atom_link_repo.derived_atom_ids(session, space.id)
    strategy = build_recall_strategy(
        payload.method, llm, session=session, embedder=embedder
    )

    # layered：主检索在 L1 子集上；flat：全量
    if payload.policy == "layered":
        l1_atoms, _l2, _l3 = partition_by_layer(
            atoms, derived_ids=derived_ids
        )
        retrieve_pool = l1_atoms if l1_atoms else atoms
    else:
        retrieve_pool = atoms

    try:
        outcome = strategy.retrieve(
            retrieve_pool, payload.query, payload.max_atoms
        )
    except RecallError as e:
        raise HTTPException(status_code=502, detail=f"recall LLM failed: {e}")
    except EmbedderError as e:
        raise HTTPException(status_code=502, detail=f"recall embedder failed: {e}")

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
        outcome.hits = hits_raw

    if payload.policy == "layered":
        hits_raw = layered_retrieve(
            atoms,
            payload.query,
            strategy_outcome=outcome,
            max_atoms=payload.max_atoms,
            derived_ids=derived_ids,
        )
    else:
        hits_raw = list(outcome.hits)

    hits_raw = _apply_derived_confidence_penalty(session, space.id, hits_raw)

    if payload.neighbor_hops >= 1:
        hits_raw = _expand_neighbors(
            session, hits_raw, max_atoms=payload.max_atoms
        )

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
    else:
        chars_used = sum(_hit_cost(h, "statement") for h in hits)
        block = render_statement_block(hits)

    hit_out = [_hit_out(h, detail=payload.detail) for h in hits]
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
