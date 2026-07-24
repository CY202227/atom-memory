"""evidence 仓储：出处链的写入与按原子汇总。"""

from typing import Iterable, Optional

from sqlmodel import Session, select

from ..models import AtomRevision, Evidence, Source


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
