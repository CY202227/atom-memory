"""atom 路由：列表 / 详情（?include=revisions,evidence）/ 展开 / 回滚 / 归档。"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session

from ...db import get_session
from ...models import Atom, AtomKind, AtomStatus, RevisionTrigger, Space, utcnow
from ...recall import render_detail_block, truncate_source
from ...repositories import atom_link_repo, atom_repo, evidence_repo, revision_repo
from .. import schemas
from ..deps import get_space, require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])

_EXPAND_EVIDENCE_MAX = 200
_INCLUDE_REVISIONS = "revisions"
_INCLUDE_EVIDENCE = "evidence"


def _get_atom_or_404(session: Session, space: Space, key: str) -> Atom:
    atom = atom_repo.get_by_key(session, space.id, key)
    if atom is None:
        raise HTTPException(status_code=404, detail="atom not found")
    return atom


def _parse_include(include: Optional[str]) -> set[str]:
    if not include or not include.strip():
        return set()
    return {part.strip().lower() for part in include.split(",") if part.strip()}


def _atom_detail(
    session: Session,
    atom: Atom,
    *,
    with_revisions: bool = False,
    with_evidence: bool = False,
) -> schemas.AtomDetail:
    dates = sorted(
        {
            f"{e['source_occurred_at']:%Y-%m-%d}"
            for e in evidence_repo.list_for_atom(session, atom.id)
        }
    )
    revisions = None
    if with_revisions:
        revisions = [
            r.model_dump(mode="json")
            for r in revision_repo.list_for_atom(session, atom.id)
        ]
    evidence = None
    if with_evidence:
        evidence = []
        for e in evidence_repo.list_for_atom(session, atom.id):
            kind = e["source_kind"]
            occurred = e["source_occurred_at"]
            evidence.append(
                {
                    "revision_seq": e["revision_seq"],
                    "change_reason": e["change_reason"],
                    "source_id": e["source_id"],
                    "source_kind": (
                        kind.value if hasattr(kind, "value") else kind
                    ),
                    "source_occurred_at": (
                        occurred.isoformat() if hasattr(occurred, "isoformat") else occurred
                    ),
                    "note": e["note"],
                }
            )
    return schemas.AtomDetail(
        id=atom.id,
        space_id=atom.space_id,
        kind=atom.kind,
        key=atom.key,
        statement=atom.statement,
        detail=atom.detail,
        happened_on=atom.happened_on,
        confidence=atom.confidence,
        memory_layer=int(atom.memory_layer or 1),
        status=atom.status,
        schema_version=atom.schema_version,
        created_at=atom.created_at,
        updated_at=atom.updated_at,
        evidence_dates=dates,
        revisions=revisions,
        evidence=evidence,
    )


@router.get("/spaces/{space_uid}/atoms", response_model=schemas.AtomListResponse)
def list_atoms(
    kind: Optional[AtomKind] = None,
    status: AtomStatus = AtomStatus.active,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    updated_after: Optional[datetime] = None,
    updated_before: Optional[datetime] = None,
    space: Space = Depends(get_space),
    session: Session = Depends(get_session),
):
    offset = (page - 1) * page_size
    items, total = atom_repo.list_atoms(
        session,
        space.id,
        kind=kind,
        status=status,
        updated_after=updated_after,
        updated_before=updated_before,
        limit=page_size,
        offset=offset,
    )
    return schemas.AtomListResponse(
        count=total,
        page=page,
        page_size=page_size,
        results=[a.model_dump(mode="json") for a in items],
    )


@router.get(
    "/spaces/{space_uid}/atoms/{key}/neighbors",
    response_model=schemas.AtomNeighborsResponse,
)
def atom_neighbors(
    key: str,
    space: Space = Depends(get_space),
    session: Session = Depends(get_session),
):
    """出边/入边（about / derived_from / contradicts），供调试与导航。"""
    atom = _get_atom_or_404(session, space, key)
    grouped = atom_link_repo.neighbors(session, atom.id)
    return schemas.AtomNeighborsResponse(
        key=atom.key,
        out=grouped.get("out") or [],
        incoming=grouped.get("in") or [],
    )


@router.post(
    "/spaces/{space_uid}/atoms/expand",
    response_model=schemas.ExpandResponse,
)
def expand_atoms(
    payload: schemas.ExpandRequest,
    space: Space = Depends(get_space),
    session: Session = Depends(get_session),
):
    hits: list[schemas.ExpandHitOut] = []
    atoms: list[Atom] = []
    missing: list[str] = []
    used = 0
    for key in payload.keys[:10]:
        atom = atom_repo.get_by_key(session, space.id, key)
        if atom is None or atom.status != AtomStatus.active:
            missing.append(key)
            continue
        detail = atom.detail or atom.statement
        cost = len(atom.statement) + len(detail)
        if hits and used + cost > payload.budget_chars:
            missing.append(key)
            continue
        excerpts = []
        if payload.with_evidence:
            from ...models import Source

            for ev in evidence_repo.list_for_atom(session, atom.id):
                src = session.get(Source, ev["source_id"])
                if src is None:
                    continue
                excerpts.append(
                    {
                        "source_id": src.id,
                        "kind": src.kind.value,
                        "excerpt": truncate_source(src.content, _EXPAND_EVIDENCE_MAX),
                    }
                )
                if len(excerpts) >= 3:
                    break
        hits.append(
            schemas.ExpandHitOut(
                key=atom.key,
                kind=atom.kind,
                statement=atom.statement,
                detail=detail,
                happened_on=atom.happened_on,
                evidence_excerpts=excerpts,
            )
        )
        atoms.append(atom)
        used += cost
    return schemas.ExpandResponse(
        hits=hits,
        missing=missing,
        context_block=render_detail_block(atoms),
    )


@router.get("/spaces/{space_uid}/atoms/{key}", response_model=schemas.AtomDetail)
def read_atom(
    key: str,
    include: Optional[str] = Query(
        default=None,
        description="逗号分隔：revisions,evidence",
    ),
    space: Space = Depends(get_space),
    session: Session = Depends(get_session),
):
    parts = _parse_include(include)
    return _atom_detail(
        session,
        _get_atom_or_404(session, space, key),
        with_revisions=_INCLUDE_REVISIONS in parts,
        with_evidence=_INCLUDE_EVIDENCE in parts,
    )


@router.post(
    "/spaces/{space_uid}/atoms/{key}/rollback",
    response_model=Atom,
    tags=["admin"],
)
def rollback_atom(
    key: str,
    payload: schemas.RollbackRequest,
    space: Space = Depends(get_space),
    session: Session = Depends(get_session),
):
    atom = _get_atom_or_404(session, space, key)
    target = revision_repo.get_by_seq(session, atom.id, payload.seq)
    if target is None:
        raise HTTPException(status_code=404, detail=f"revision seq={payload.seq} not found")
    atom.statement = target.statement
    atom.detail = target.detail
    atom.happened_on = target.happened_on
    atom.status = AtomStatus.active
    atom.updated_at = utcnow()
    revision_repo.add(
        session, atom, f"回滚到第 {payload.seq} 版", RevisionTrigger.rollback
    )
    session.commit()
    session.refresh(atom)
    return atom


@router.post("/spaces/{space_uid}/atoms/{key}/archive", response_model=Atom)
def archive_atom(
    key: str,
    space: Space = Depends(get_space),
    session: Session = Depends(get_session),
):
    atom = _get_atom_or_404(session, space, key)
    if atom.status == AtomStatus.archived:
        return atom
    atom.status = AtomStatus.archived
    atom.updated_at = utcnow()
    revision_repo.add(session, atom, "人工归档（用户要求遗忘）", RevisionTrigger.manual)
    session.commit()
    session.refresh(atom)
    return atom
