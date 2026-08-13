"""atom_embedding 仓储：读缓存 / 写回 / 失效。"""

from sqlmodel import Session, select

from ..models import AtomEmbedding, utcnow


def get_for_atoms(
    session: Session, atom_ids: list[int]
) -> dict[int, AtomEmbedding]:
    if not atom_ids:
        return {}
    rows = session.exec(
        select(AtomEmbedding).where(AtomEmbedding.atom_id.in_(atom_ids))
    ).all()
    return {r.atom_id: r for r in rows}


def upsert(
    session: Session,
    *,
    atom_id: int,
    space_id: int,
    model_tag: str,
    vector: list[float],
    existing: AtomEmbedding | None = None,
) -> AtomEmbedding:
    if existing is None:
        existing = session.exec(
            select(AtomEmbedding).where(AtomEmbedding.atom_id == atom_id)
        ).first()
    if existing is None:
        row = AtomEmbedding(
            atom_id=atom_id,
            space_id=space_id,
            model_tag=model_tag,
            dim=len(vector),
            vector=vector,
        )
        session.add(row)
        return row
    existing.space_id = space_id
    existing.model_tag = model_tag
    existing.dim = len(vector)
    existing.vector = vector
    existing.updated_at = utcnow()
    session.add(existing)
    return existing


def invalidate_for_atom(session: Session, atom_id: int) -> None:
    row = session.exec(
        select(AtomEmbedding).where(AtomEmbedding.atom_id == atom_id)
    ).first()
    if row is not None:
        session.delete(row)


def clear_for_space(session: Session, space_id: int) -> None:
    rows = session.exec(
        select(AtomEmbedding).where(AtomEmbedding.space_id == space_id)
    ).all()
    for row in rows:
        session.delete(row)
