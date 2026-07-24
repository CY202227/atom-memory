"""atom 路由：索引 / 列表 / 详情 / 展开 / 修订 / 出处 / 回滚 / 归档。"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from ...db import get_session
from ...models import Atom, AtomKind, AtomRevision, AtomStatus, RevisionTrigger, Space, utcnow
from ...recall import render_detail_block, truncate_source
from ...repositories import atom_repo, evidence_repo, revision_repo
from .. import schemas
from ..deps import get_space, require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])

_EXPAND_EVIDENCE_MAX = 200


def _get_atom_or_404(session: Session, space: Space, key: str) -> Atom:
    atom = atom_repo.get_by_key(session, space.id, key)
    if atom is None:
        raise HTTPException(status_code=404, detail="atom not found")
    return atom


def _atom_detail(session: Session, atom: Atom) -> schemas.AtomDetail:
    dates = sorted(
        {
            f"{e['source_occurred_at']:%Y-%m-%d}"
            for e in evidence_repo.list_for_atom(session, atom.id)
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
        status=atom.status,
        schema_version=atom.schema_version,
        created_at=atom.created_at,
        updated_at=atom.updated_at,
        evidence_dates=dates,
    )


@router.get("/spaces/{space_uid}/index", response_model=list[schemas.IndexEntry])
def read_index(space: Space = Depends(get_space), session: Session = Depends(get_session)):
    return [
        schemas.IndexEntry(
            kind=a.kind,
            key=a.key,
            statement=a.statement,
            updated_at=a.updated_at,
        )
        for a in atom_repo.list_active(session, space.id)
    ]


@router.get("/spaces/{space_uid}/atoms", response_model=list[Atom])
def list_atoms(
    kind: Optional[AtomKind] = None,
    status: AtomStatus = AtomStatus.active,
    space: Space = Depends(get_space),
    session: Session = Depends(get_session),
):
    return atom_repo.list_atoms(session, space.id, kind=kind, status=status)


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
    key: str, space: Space = Depends(get_space), session: Session = Depends(get_session)
):
    return _atom_detail(session, _get_atom_or_404(session, space, key))


@router.get(
    "/spaces/{space_uid}/atoms/{key}/revisions",
    response_model=list[AtomRevision],
)
def list_revisions(
    key: str, space: Space = Depends(get_space), session: Session = Depends(get_session)
):
    atom = _get_atom_or_404(session, space, key)
    return revision_repo.list_for_atom(session, atom.id)


@router.get("/spaces/{space_uid}/atoms/{key}/evidence")
def list_evidence(
    key: str, space: Space = Depends(get_space), session: Session = Depends(get_session)
):
    atom = _get_atom_or_404(session, space, key)
    return evidence_repo.list_for_atom(session, atom.id)


@router.post("/spaces/{space_uid}/atoms/{key}/rollback", response_model=Atom)
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
