"""atom 仓储：原子认识的查找与列举。"""

from typing import Optional

from sqlmodel import Session, select

from ..models import Atom, AtomKind, AtomStatus


def get_by_key(session: Session, space_id: int, key: str) -> Optional[Atom]:
    return session.exec(
        select(Atom).where(Atom.space_id == space_id, Atom.key == key)
    ).first()


def list_active(session: Session, space_id: int) -> list[Atom]:
    return list(
        session.exec(
            select(Atom)
            .where(Atom.space_id == space_id, Atom.status == AtomStatus.active)
            .order_by(Atom.kind, Atom.key)
        ).all()
    )


def list_atoms(
    session: Session,
    space_id: int,
    kind: Optional[AtomKind] = None,
    status: AtomStatus = AtomStatus.active,
) -> list[Atom]:
    stmt = select(Atom).where(Atom.space_id == space_id, Atom.status == status)
    if kind:
        stmt = stmt.where(Atom.kind == kind)
    return list(session.exec(stmt.order_by(Atom.updated_at.desc())).all())
