"""Invalidate cached retrieval and derived facts after any atom mutation."""
from sqlmodel import Session
from .models import Atom, utcnow
from .repositories import atom_link_repo, embedding_repo


def invalidate_dependents(session: Session, atom_id: int) -> None:
    for dependent_id in atom_link_repo.reverse_derived_closure(session, {atom_id}):
        child = session.get(Atom, dependent_id)
        if child is None:
            continue
        child.confidence = min(float(child.confidence), 0.3) * 0.5 if child.confidence is not None else 0.2
        child.updated_at = utcnow()
        session.add(child)
        embedding_repo.invalidate_for_atom(session, dependent_id)


def invalidate_atom(session: Session, atom_id: int) -> None:
    embedding_repo.invalidate_for_atom(session, atom_id)
    invalidate_dependents(session, atom_id)
