"""按上游来源删除（delete-by-source）：隐私删除语义，Atom-first。"""

import json
from dataclasses import dataclass, field

from sqlmodel import Session, select

from .consolidation import prompts
from .consolidation.engine import parse_json_object
from .llm.base import ChatLLM, LLMError
from .models import (
    Atom,
    AtomRevision,
    ConsolidationRun,
    Evidence,
    RevisionTrigger,
    RunStatus,
    Source,
    Space,
    utcnow,
)
from .repositories import (
    atom_link_repo,
    evidence_repo,
    embedding_repo,
    revision_repo,
    run_repo,
)


class RedactionError(Exception):
    """重固化失败：整个删除请求原子回退。"""


@dataclass
class DeletionImpact:
    sources: list[Source]
    atoms_to_delete: list[Atom]
    atoms_to_reconsolidate: list[Atom]
    atoms_derived_affected: list[Atom] = field(default_factory=list)
    remaining_by_atom: dict[int, set[int]] = field(default_factory=dict)

    @property
    def source_ids(self) -> set[int]:
        return {s.id for s in self.sources}


def match_sources(session: Session, space_id: int, ref: dict) -> list[Source]:
    sources = session.exec(select(Source).where(Source.space_id == space_id)).all()
    return [
        s
        for s in sources
        if s.external_ref and all(s.external_ref.get(k) == v for k, v in ref.items())
    ]


def assess(session: Session, space_id: int, ref: dict) -> DeletionImpact:
    sources = match_sources(session, space_id, ref)
    impact = DeletionImpact(
        sources=sources, atoms_to_delete=[], atoms_to_reconsolidate=[]
    )
    if not sources:
        return impact

    rows = session.exec(
        select(Atom, Evidence.source_id)
        .where(Atom.space_id == space_id)
        .where(AtomRevision.atom_id == Atom.id)
        .where(Evidence.revision_id == AtomRevision.id)
    ).all()
    evidence_by_atom: dict[int, set[int]] = {}
    atom_by_id: dict[int, Atom] = {}
    for atom, source_id in rows:
        atom_by_id[atom.id] = atom
        evidence_by_atom.setdefault(atom.id, set()).add(source_id)

    matched = impact.source_ids
    for atom_id, cited in sorted(evidence_by_atom.items()):
        if not (cited & matched):
            continue
        remaining = cited - matched
        if remaining:
            impact.atoms_to_reconsolidate.append(atom_by_id[atom_id])
            impact.remaining_by_atom[atom_id] = remaining
        else:
            impact.atoms_to_delete.append(atom_by_id[atom_id])

    seed_ids = {
        a.id for a in impact.atoms_to_delete + impact.atoms_to_reconsolidate
        if a.id is not None
    }
    derived_ids = atom_link_repo.reverse_derived_closure(session, seed_ids)
    # 已在 delete/recon 集合中的不算「仅派生受影响」
    direct_ids = seed_ids
    for did in sorted(derived_ids - direct_ids):
        atom = atom_by_id.get(did) or session.get(Atom, did)
        if atom is not None:
            atom_by_id[atom.id] = atom
            impact.atoms_derived_affected.append(atom)
    return impact


def execute(session: Session, space: Space, ref: dict, llm: ChatLLM) -> dict:
    impact = assess(session, space.id, ref)
    if not impact.sources:
        return {
            "deleted_sources": 0,
            "deleted_atoms": [],
            "reconsolidated_atoms": [],
            "derived_affected_atoms": [],
            "run_id": None,
        }

    run = run_repo.create(
        session, space.id, "delete_by_source", sorted(impact.source_ids)
    )
    try:
        rewrites = _redact_all(session, llm, run, impact)
    except (LLMError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        run_repo.finish(session, run, RunStatus.failed, error=f"{type(e).__name__}: {e}")
        raise RedactionError(str(e)) from e

    _apply(session, run, impact, rewrites)
    deleted = [a.key for a in impact.atoms_to_delete]
    recon = [a.key for a in impact.atoms_to_reconsolidate]
    derived = [a.key for a in impact.atoms_derived_affected]
    run_repo.finish(
        session,
        run,
        RunStatus.succeeded,
        atoms_touched=deleted + recon + derived,
    )
    return {
        "deleted_sources": len(impact.sources),
        "deleted_atoms": deleted,
        "reconsolidated_atoms": recon,
        "derived_affected_atoms": derived,
        "run_id": run.id,
    }


def _redact_all(
    session: Session, llm: ChatLLM, run: ConsolidationRun, impact: DeletionImpact
) -> dict[int, dict]:
    rewrites: dict[int, dict] = {}
    for atom in impact.atoms_to_reconsolidate:
        remaining = list(
            session.exec(
                select(Source).where(Source.id.in_(impact.remaining_by_atom[atom.id]))
            ).all()
        )
        res = llm.complete(
            prompts.REDACT_SYSTEM,
            f"## 原子当前全文\n{prompts.render_atoms_full([atom])}\n\n"
            f"## 剩余证据材料\n{prompts.render_sources(remaining)}",
        )
        run.prompt_tokens += res.prompt_tokens
        run.completion_tokens += res.completion_tokens
        rewrite = parse_json_object(res.text)
        if not rewrite.get("statement"):
            raise ValueError(f"redact output for atom {atom.key!r} has no statement")
        rewrites[atom.id] = rewrite
    return rewrites


def _stale_derived(session: Session, atom: Atom) -> None:
    """派生 atom 失效：降置信度 + 清向量。"""
    if atom.confidence is not None:
        atom.confidence = min(float(atom.confidence), 0.3) * 0.5
    else:
        atom.confidence = 0.2
    atom.updated_at = utcnow()
    if atom.id is not None:
        embedding_repo.invalidate_for_atom(session, atom.id)
        session.add(atom)


def _apply(
    session: Session,
    run: ConsolidationRun,
    impact: DeletionImpact,
    rewrites: dict[int, dict],
) -> None:
    matched = impact.source_ids

    for ev in session.exec(select(Evidence).where(Evidence.source_id.in_(matched))).all():
        session.delete(ev)

    for atom in impact.atoms_to_delete:
        if atom.id is not None:
            atom_link_repo.delete_for_atom(session, atom.id)
            embedding_repo.invalidate_for_atom(session, atom.id)
        for rev in session.exec(
            select(AtomRevision).where(AtomRevision.atom_id == atom.id)
        ).all():
            session.delete(rev)
        session.delete(atom)

    for atom in impact.atoms_to_reconsolidate:
        rewrite = rewrites[atom.id]
        atom.statement = (rewrite.get("statement") or atom.statement)[:80]
        atom.detail = (rewrite.get("detail") or "")[:300]
        if rewrite.get("confidence") is not None:
            atom.confidence = rewrite["confidence"]
        atom.updated_at = utcnow()
        if atom.id is not None:
            embedding_repo.invalidate_for_atom(session, atom.id)
        rev = revision_repo.add(
            session,
            atom,
            "来源隐私删除后重固化：剔除被删证据，按剩余材料重写",
            RevisionTrigger.consolidation,
            run_id=run.id,
        )
        evidence_repo.add_many(
            session, rev.id, sorted(impact.remaining_by_atom[atom.id])
        )

    for atom in impact.atoms_derived_affected:
        _stale_derived(session, atom)

    for source in impact.sources:
        session.delete(source)

    session.commit()
