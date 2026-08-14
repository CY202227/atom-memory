"""L3 Persona 收敛：从 L1/L2 收敛稳定画像与称呼（独立于 consolidate）。"""

from __future__ import annotations

from datetime import date

from sqlmodel import Session

from ..linking import keyify
from ..llm.base import ChatLLM, LLMError
from ..memory_layers import LAYER_L3
from ..models import (
    Atom,
    AtomKind,
    AtomLinkKind,
    AtomStatus,
    ConsolidationRun,
    RevisionTrigger,
    RunStatus,
    Space,
    utcnow,
)
from ..repositories import (
    atom_link_repo,
    atom_repo,
    embedding_repo,
    evidence_repo,
    revision_repo,
    run_repo,
)
from . import prompts
from .canonical import CANONICAL_PERSONA, CANONICAL_PREFERRED_NAME
from .engine import _DETAIL_MAX, _STATEMENT_MAX, parse_json_object

_MAX_PERSONA_OPS = 4


class PersonaEngine:
    def __init__(self, llm: ChatLLM | None = None):
        self._llm = llm

    def run(
        self,
        session: Session,
        space: Space,
        *,
        trigger: str = "persona",
        max_read: int = 12,
    ) -> ConsolidationRun:
        active = run_repo.find_active(session, space.id, stale_seconds=1800)
        if active is not None:
            return active

        atoms = atom_repo.list_active(session, space.id)
        run = run_repo.create(session, space.id, trigger, [])
        try:
            touched = self._converge(
                session, space, run, atoms, max_read=max_read
            )
            return run_repo.finish(
                session, run, RunStatus.succeeded, atoms_touched=touched
            )
        except (LLMError, ValueError, KeyError, TypeError) as e:
            session.rollback()
            run = session.get(ConsolidationRun, run.id) or run
            touched = self._mark_existing_l3(session, run)
            session.commit()
            return run_repo.finish(
                session,
                run,
                RunStatus.succeeded,
                atoms_touched=touched,
                error=f"persona_degraded: {type(e).__name__}: {e}",
            )

    def _mark_existing_l3(
        self, session: Session, run: ConsolidationRun
    ) -> list[str]:
        touched: list[str] = []
        atoms = atom_repo.list_active(session, run.space_id)
        for atom in atoms:
            if atom.key in (CANONICAL_PERSONA, CANONICAL_PREFERRED_NAME) or (
                atom.kind == AtomKind.self
            ):
                if atom.memory_layer != LAYER_L3:
                    atom.memory_layer = LAYER_L3
                    atom.updated_at = utcnow()
                    session.add(atom)
                    revision_repo.add(
                        session,
                        atom,
                        "标为 L3 persona",
                        RevisionTrigger.consolidation,
                        run_id=run.id,
                    )
                    touched.append(atom.key)
        return touched

    def _converge(
        self,
        session: Session,
        space: Space,
        run: ConsolidationRun,
        atoms: list[Atom],
        *,
        max_read: int,
    ) -> list[str]:
        if not atoms:
            return []

        if self._llm is None:
            touched = self._mark_existing_l3(session, run)
            session.commit()
            return touched

        candidates = [
            a
            for a in atoms
            if a.kind
            in (AtomKind.self, AtomKind.person, AtomKind.belief)
            or a.memory_layer in (2, 3)
            or a.key in (CANONICAL_PERSONA, CANONICAL_PREFERRED_NAME)
        ]
        if not candidates:
            touched = self._mark_existing_l3(session, run)
            session.commit()
            return touched

        index_text = prompts.render_index(candidates)
        context = candidates[:max_read]
        res = self._llm.complete(
            prompts.PERSONA_SYSTEM,
            f"## 当前时间\n{date.today().isoformat()}\n\n"
            f"## atom 索引\n{index_text}\n\n"
            f"## 相关原子全文\n{prompts.render_atoms_full(context)}",
        )
        run.prompt_tokens += res.prompt_tokens
        run.completion_tokens += res.completion_tokens
        operations = parse_json_object(res.text).get("operations") or []

        by_key = {a.key: a for a in atoms}
        touched: list[str] = []
        for op in operations[:_MAX_PERSONA_OPS]:
            key = self._apply_persona_op(session, space, run, op, by_key)
            if key:
                touched.append(key)
                by_key = {
                    a.key: a for a in atom_repo.list_active(session, space.id)
                }

        for k in self._mark_existing_l3(session, run):
            if k not in touched:
                touched.append(k)
        session.commit()
        return touched

    def _apply_persona_op(
        self,
        session: Session,
        space: Space,
        run: ConsolidationRun,
        op: dict,
        by_key: dict[str, Atom],
    ) -> str | None:
        if op.get("op") != "upsert":
            return None
        raw_key = keyify(op.get("key") or "")
        if raw_key not in (CANONICAL_PERSONA, CANONICAL_PREFERRED_NAME):
            kind_hint = str(op.get("kind") or "")
            if kind_hint == AtomKind.person.value or "称呼" in (
                op.get("change_reason") or ""
            ):
                raw_key = CANONICAL_PREFERRED_NAME
            else:
                raw_key = CANONICAL_PERSONA
        key = raw_key
        statement = (op.get("statement") or "").strip()[:_STATEMENT_MAX]
        if not statement:
            return None

        derived_keys = [
            keyify(str(k))
            for k in (op.get("derived_from") or [])
            if keyify(str(k))
        ]
        parents = [
            by_key[k] for k in derived_keys if k in by_key and by_key[k].id
        ]
        detail = (op.get("detail") or "").strip()[:_DETAIL_MAX]
        conf = op.get("confidence")
        try:
            confidence = float(conf) if conf is not None else 0.9
        except (TypeError, ValueError):
            confidence = 0.9
        confidence = max(0.0, min(1.0, confidence))

        atom_kind = (
            AtomKind.person
            if key == CANONICAL_PREFERRED_NAME
            else AtomKind.self
        )
        atom = atom_repo.get_by_key(session, space.id, key)
        if atom is None:
            atom = Atom(
                space_id=space.id,
                kind=atom_kind,
                key=key,
                statement=statement,
                detail=detail,
                confidence=confidence,
                memory_layer=LAYER_L3,
            )
            session.add(atom)
            session.flush()
            reason = op.get("change_reason") or "persona 收敛"
        else:
            atom.kind = atom_kind
            atom.statement = statement
            atom.detail = detail
            atom.confidence = confidence
            atom.memory_layer = LAYER_L3
            atom.status = AtomStatus.active
            atom.updated_at = utcnow()
            session.add(atom)
            reason = op.get("change_reason") or "persona 更新"

        source_ids: set[int] = set()
        for parent in parents:
            for ev in evidence_repo.list_for_atom(session, parent.id):
                source_ids.add(int(ev["source_id"]))
        if not source_ids and atom.id is not None:
            for ev in evidence_repo.list_for_atom(session, atom.id):
                source_ids.add(int(ev["source_id"]))

        rev = revision_repo.add(
            session, atom, reason, RevisionTrigger.consolidation, run_id=run.id
        )
        if source_ids:
            evidence_repo.add_many(session, rev.id, sorted(source_ids))

        if atom.id is not None:
            embedding_repo.invalidate_for_atom(session, atom.id)
            if parents:
                links = [
                    (p.id, AtomLinkKind.derived_from, None)
                    for p in parents
                    if p.id is not None
                ]
                atom_link_repo.replace_out_links(
                    session,
                    space_id=space.id,
                    src_atom_id=atom.id,
                    links=links,
                )
        return key
