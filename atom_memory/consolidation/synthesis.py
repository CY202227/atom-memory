"""综合层：从已有 atom 归纳更高层认识（独立于 consolidate）。"""

from __future__ import annotations

from datetime import date

from sqlmodel import Session

from ..llm.base import ChatLLM, LLMError
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
from ..memory_layers import LAYER_L2
from . import prompts
from .engine import (
    _DETAIL_MAX,
    _STATEMENT_MAX,
    _json_schema_format,
    parse_json_object,
)

_MAX_CONFIDENCE = 0.8


class SynthesisEngine:
    def __init__(self, llm: ChatLLM):
        self._llm = llm

    def run(
        self,
        session: Session,
        space: Space,
        *,
        trigger: str = "synthesize",
        max_read: int = 8,
    ) -> ConsolidationRun:
        active = run_repo.find_active(session, space.id, stale_seconds=1800)
        if active is not None:
            return active

        atoms = atom_repo.list_active(session, space.id)
        run = run_repo.create(session, space.id, trigger, [])
        if len(atoms) < 2:
            return run_repo.finish(session, run, RunStatus.succeeded, atoms_touched=[])

        try:
            touched = self._synthesize(session, space, run, atoms, max_read=max_read)
            return run_repo.finish(
                session, run, RunStatus.succeeded, atoms_touched=touched
            )
        except (LLMError, ValueError, KeyError, TypeError) as e:
            session.rollback()
            return run_repo.finish(
                session, run, RunStatus.failed, error=f"{type(e).__name__}: {e}"
            )

    def _synthesize(
        self,
        session: Session,
        space: Space,
        run: ConsolidationRun,
        atoms: list[Atom],
        *,
        max_read: int,
    ) -> list[str]:
        index_text = prompts.render_index(atoms)
        res = self._llm.complete(
            prompts.SYNTHESIZE_SELECT_SYSTEM,
            f"## atom 索引\n{index_text}",
        )
        run.prompt_tokens += res.prompt_tokens
        run.completion_tokens += res.completion_tokens
        read_keys = parse_json_object(res.text).get("read") or []
        by_key = {a.key: a for a in atoms}
        context = [by_key[k] for k in read_keys if k in by_key][:max_read]
        if len(context) < 2:
            # 选不够时取索引前若干条兜底
            context = atoms[: min(max_read, len(atoms))]
        if len(context) < 2:
            return []

        res = self._llm.complete(
            prompts.SYNTHESIZE_SYSTEM,
            f"## 当前时间\n{date.today().isoformat()}\n\n"
            f"## atom 索引\n{index_text}\n\n"
            f"## 相关原子全文\n{prompts.render_atoms_full(context)}",
        )
        run.prompt_tokens += res.prompt_tokens
        run.completion_tokens += res.completion_tokens
        operations = parse_json_object(res.text).get("operations") or []

        touched: list[str] = []
        for op in operations:
            key = self._apply_synth_op(session, space, run, op, by_key)
            if key:
                touched.append(key)
                by_key = {a.key: a for a in atom_repo.list_active(session, space.id)}
        session.commit()
        return touched

    def _apply_synth_op(
        self,
        session: Session,
        space: Space,
        run: ConsolidationRun,
        op: dict,
        by_key: dict[str, Atom],
    ) -> str | None:
        if op.get("op") != "upsert":
            return None
        from ..linking import keyify

        key = keyify(op.get("key") or "")
        statement = (op.get("statement") or "").strip()[:_STATEMENT_MAX]
        if not key or not statement:
            return None

        derived_keys = [
            keyify(str(k))
            for k in (op.get("derived_from") or [])
            if keyify(str(k))
        ]
        # 去重保序
        seen: set[str] = set()
        derived_keys = [k for k in derived_keys if not (k in seen or seen.add(k))]
        parents = [by_key[k] for k in derived_keys if k in by_key and by_key[k].id]
        if len(parents) < 2:
            return None

        conf = op.get("confidence")
        if conf is None:
            return None
        try:
            confidence = min(float(conf), _MAX_CONFIDENCE)
        except (TypeError, ValueError):
            return None
        if confidence < 0:
            return None

        parent_keys = [p.key for p in parents]
        detail = (op.get("detail") or "").strip()
        suffix = f" | 推自：{', '.join(parent_keys)}"
        if "推自：" not in detail:
            room = max(0, _DETAIL_MAX - len(suffix))
            detail = (detail[:room].rstrip() + suffix)[:_DETAIL_MAX]
        else:
            detail = detail[:_DETAIL_MAX]

        reason = op.get("change_reason") or f"综合自 {', '.join(parent_keys)}"
        atom_kind = AtomKind.belief
        try:
            if op.get("kind"):
                atom_kind = AtomKind(op["kind"])
        except ValueError:
            atom_kind = AtomKind.belief

        atom = atom_repo.get_by_key(session, space.id, key)
        if atom is None:
            atom = Atom(
                space_id=space.id,
                kind=atom_kind,
                key=key,
                statement=statement,
                detail=detail,
                confidence=confidence,
                memory_layer=LAYER_L2,
            )
            session.add(atom)
            session.flush()
        else:
            atom.kind = atom_kind
            atom.statement = statement
            atom.detail = detail
            atom.confidence = confidence
            # 综合层产物标 L2；不覆盖已有 L3 画像
            if atom.memory_layer != 3:
                atom.memory_layer = LAYER_L2
            atom.status = AtomStatus.active
            atom.updated_at = utcnow()
            session.add(atom)

        # evidence = 父 atom 全部 source 并集
        source_ids: set[int] = set()
        for parent in parents:
            for ev in evidence_repo.list_for_atom(session, parent.id):
                source_ids.add(int(ev["source_id"]))

        rev = revision_repo.add(
            session, atom, reason, RevisionTrigger.consolidation, run_id=run.id
        )
        if source_ids:
            evidence_repo.add_many(session, rev.id, sorted(source_ids))

        if atom.id is not None:
            embedding_repo.invalidate_for_atom(session, atom.id)
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
