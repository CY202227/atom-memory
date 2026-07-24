"""space 仓储：记忆库的创建与查找。"""

from typing import Optional

from sqlmodel import Session, select

from ..models import Space


def get_by_uid(session: Session, uid: str) -> Optional[Space]:
    return session.exec(select(Space).where(Space.uid == uid)).first()


def get_or_create(
    session: Session,
    uid: Optional[str] = None,
    owner_id: Optional[str] = None,
    subject_id: Optional[str] = None,
) -> Space:
    if uid:
        existing = get_by_uid(session, uid)
        if existing:
            return existing
    elif owner_id and subject_id:
        existing = session.exec(
            select(Space).where(
                Space.owner_id == owner_id, Space.subject_id == subject_id
            )
        ).first()
        if existing:
            return existing
    space = Space(owner_id=owner_id, subject_id=subject_id)
    if uid:
        space.uid = uid
    session.add(space)
    session.commit()
    session.refresh(space)
    return space


def list_spaces(session: Session, owner_id: Optional[str] = None) -> list[Space]:
    stmt = select(Space)
    if owner_id:
        stmt = stmt.where(Space.owner_id == owner_id)
    return list(session.exec(stmt.order_by(Space.created_at)).all())


def delete_space(session: Session, space: Space) -> dict:
    """整库删除某 space：级联清掉其全部行，返回删除计数。"""
    from ..models import Atom, AtomRevision, ConsolidationRun, Evidence, Source

    atoms = session.exec(select(Atom).where(Atom.space_id == space.id)).all()
    atom_ids = [a.id for a in atoms]
    revs = (
        session.exec(
            select(AtomRevision).where(AtomRevision.atom_id.in_(atom_ids))
        ).all()
        if atom_ids
        else []
    )
    rev_ids = [r.id for r in revs]
    counts = {"atoms": len(atoms), "revisions": len(revs), "sources": 0}
    if rev_ids:
        for ev in session.exec(select(Evidence).where(Evidence.revision_id.in_(rev_ids))).all():
            session.delete(ev)
    for r in revs:
        session.delete(r)
    for a in atoms:
        session.delete(a)
    for s in session.exec(select(Source).where(Source.space_id == space.id)).all():
        counts["sources"] += 1
        session.delete(s)
    for run in session.exec(
        select(ConsolidationRun).where(ConsolidationRun.space_id == space.id)
    ).all():
        session.delete(run)
    session.delete(space)
    session.commit()
    return counts
