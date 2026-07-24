"""atom_revision 仓储：修订历史的追加与读取。"""

from typing import Optional

from sqlmodel import Session, select

from ..models import Atom, AtomRevision, RevisionTrigger


def list_for_atom(session: Session, atom_id: int) -> list[AtomRevision]:
    return list(
        session.exec(
            select(AtomRevision)
            .where(AtomRevision.atom_id == atom_id)
            .order_by(AtomRevision.seq)
        ).all()
    )


def get_by_seq(session: Session, atom_id: int, seq: int) -> Optional[AtomRevision]:
    return session.exec(
        select(AtomRevision).where(
            AtomRevision.atom_id == atom_id, AtomRevision.seq == seq
        )
    ).first()


def add(
    session: Session,
    atom: Atom,
    change_reason: str,
    trigger: RevisionTrigger,
    run_id: Optional[int] = None,
) -> AtomRevision:
    last = session.exec(
        select(AtomRevision)
        .where(AtomRevision.atom_id == atom.id)
        .order_by(AtomRevision.seq.desc())
        .limit(1)
    ).first()
    rev = AtomRevision(
        atom_id=atom.id,
        seq=(last.seq if last else 0) + 1,
        statement=atom.statement,
        detail=atom.detail,
        happened_on=atom.happened_on,
        change_reason=change_reason,
        trigger=trigger,
        run_id=run_id,
    )
    session.add(rev)
    session.flush()
    return rev
