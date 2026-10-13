"""evidence 仓储：出处链的写入与按原子汇总。"""

from typing import Iterable, Optional

from sqlmodel import Session, select

from ..models import Atom, AtomRevision, Evidence, Source


def add_many(
    session: Session,
    revision_id: int,
    source_ids: Iterable[int],
    note: Optional[str] = None,
) -> None:
    for sid in source_ids:
        session.add(Evidence(revision_id=revision_id, source_id=sid, note=note))


def list_for_atom(session: Session, atom_id: int) -> list[dict]:
    rows = session.exec(
        select(AtomRevision, Evidence, Source)
        .where(AtomRevision.atom_id == atom_id)
        .where(Evidence.revision_id == AtomRevision.id)
        .where(Source.id == Evidence.source_id)
        .order_by(AtomRevision.seq)
    ).all()
    return [
        {
            "revision_seq": rev.seq,
            "change_reason": rev.change_reason,
            "source_id": src.id,
            "source_kind": src.kind,
            "source_occurred_at": src.occurred_at,
            "note": ev.note,
        }
        for rev, ev, src in rows
    ]


def current_sources(session: Session, atom: Atom) -> list[Source]:
    """Use the latest revision's receipts; identical text does not imply identical source time."""
    revision = session.exec(select(AtomRevision).where(AtomRevision.atom_id == atom.id)
        .order_by(AtomRevision.seq.desc())).first()
    if revision is None or any(getattr(revision, field) != getattr(atom, field)
            for field in ("statement", "detail", "happened_on")):
        return []
    return sources_for_revision(session, revision.id, atom.space_id)


def sources_for_revision(session: Session, revision_id: int, space_id: int) -> list[Source]:
    rows = session.exec(select(Source).where(Evidence.revision_id == revision_id)
        .where(Source.id == Evidence.source_id).where(Source.space_id == space_id)).all()
    return list({source.id: source for source in rows}.values())
