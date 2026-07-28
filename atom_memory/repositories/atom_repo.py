"""atom 仓储：原子认识的查找与列举。"""

from datetime import datetime
from typing import Optional

from sqlmodel import Session, col, func, select

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
    *,
    updated_after: datetime | None = None,
    updated_before: datetime | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> tuple[list[Atom], int]:
    """按更新时间倒序列举；返回 (本页 items, 匹配总数)。"""
    filters = [Atom.space_id == space_id, Atom.status == status]
    if kind:
        filters.append(Atom.kind == kind)
    if updated_after is not None:
        filters.append(Atom.updated_at >= updated_after)
    if updated_before is not None:
        filters.append(Atom.updated_at <= updated_before)

    total = session.exec(
        select(func.count()).select_from(Atom).where(*filters)
    ).one()

    stmt = (
        select(Atom)
        .where(*filters)
        .order_by(col(Atom.updated_at).desc())
    )
    if offset:
        stmt = stmt.offset(offset)
    if limit is not None:
        stmt = stmt.limit(limit)
    items = list(session.exec(stmt).all())
    return items, int(total)
