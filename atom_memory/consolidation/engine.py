"""固化引擎：Atom-first，wiki 唯一的常规写通路。

两阶段：select（选 key 读 detail）→ write（产出 upsert/archive）。
"""

import json
import copy
import re
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field
from sqlmodel import Session

from ..linking import keyify
from ..llm.base import ChatLLM, LLMError
from ..models import (
    Atom,
    AtomKind,
    AtomStatus,
    ConsolidationRun,
    RevisionTrigger,
    RunStatus,
    Source,
    Space,
    utcnow,
)
from ..models import AtomLinkKind
from ..repositories import (
    atom_link_repo,
    atom_repo,
    embedding_repo,
    evidence_repo,
    revision_repo,
    run_repo,
    source_repo,
)
from ..memory_layers import default_layer_for_write
from ..temporal import append_time_facts_to_detail, source_time_anchor
from . import prompts
from .canonical import apply_canonical_to_op

_STATEMENT_MAX = 80
_DETAIL_MAX = 300


def parse_json_object(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            value = json.loads(text[start : end + 1])
        else:
            raise
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


class AtomLinkOp(BaseModel):
    to: str
    kind: Literal["about", "contradicts", "derived_from"] = "about"


class AtomOp(BaseModel):
    op: Literal["upsert", "archive"]
    kind: AtomKind = AtomKind.belief
    key: str
    statement: str = ""
    detail: str = ""
    happened_on: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    change_reason: str = ""
    source_ids: list[int] = []
    links: list[AtomLinkOp] = []
    memory_layer: Literal[1, 2, 3] | None = None


class WritePlan(BaseModel):
    operations: list[AtomOp]


class SelectPlan(BaseModel):
    read: list[str]


def _json_schema_format(name: str, model: type[BaseModel]) -> dict:
    schema = copy.deepcopy(model.model_json_schema())
    def strict(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in node.values():
                strict(value)
        elif isinstance(node, list):
            for value in node:
                strict(value)
    strict(schema)
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }


def request_plan(llm: ChatLLM, system: str, user: str, model: type[BaseModel], name: str,
                 run: ConsolidationRun | None = None) -> BaseModel:
    """Parse and validate before any write; retry schema violations once without raw output."""
    for attempt in range(2):
        result = llm.complete(system, user, response_format=_json_schema_format(name, model))
        if run is not None:
            run.prompt_tokens += result.prompt_tokens
            run.completion_tokens += result.completion_tokens
        try:
            return model.model_validate(parse_json_object(result.text))
        except ValueError as exc:
            if attempt:
                raise LLMError("Structured output failed schema validation after 2 attempts", code="invalid_schema") from exc
            user += "\nReturn a complete JSON object matching every field and type in the schema. The previous output failed schema validation; do not omit the required top-level fields."
    raise AssertionError("unreachable")


def _parse_happened_on(value) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def resolve_happened_on(
    value,
    *,
    source_texts: str,
    existing: date | None,
    today: date | None = None,
) -> date | None:
    """解析 happened_on：臆造年份丢弃；LLM 给 null 时保留已有日期。

    年份须出现在引用材料中，或等于「当前时间」年份；否则视为幻觉。
    若材料已支撑旧日期、却不支撑新日期，保留已有（防「今天」覆盖纠正）。
    """
    today = today or date.today()
    texts = source_texts or ""
    parsed = _parse_happened_on(value)
    if parsed is None:
        return existing
    year_s = str(parsed.year)
    if year_s not in texts and parsed.year != today.year:
        return existing
    if existing is not None and parsed != existing:
        if _date_grounded(existing, texts, today) and not _date_grounded(
            parsed, texts, today
        ):
            return existing
    return parsed


def _date_grounded(d: date, texts: str, today: date) -> bool:
    """材料是否明确支撑该公历日（含「今天/昨天」相对表述）。"""
    if d.isoformat() in texts:
        return True
    if f"{d.month}月{d.day}日" in texts or f"{d.month}月{d.day:02d}日" in texts:
        return True
    if f"{d.month:02d}-{d.day:02d}" in texts or f"{d.month}-{d.day}" in texts:
        return True
    if d == today and _mentions_relative_today(texts):
        return True
    if d.toordinal() == today.toordinal() - 1 and (
        "昨天" in texts or "昨日" in texts
    ):
        return True
    return False


def _mentions_relative_today(texts: str) -> bool:
    """「今天/今日」肯定提及；排除「不是今天」等否定。"""
    cleaned = re.sub(r"不[是]?今天|并非今天|不是今日|并非今日", "", texts)
    return "今天" in cleaned or "今日" in cleaned


class ConsolidationEngine:
    def __init__(self, llm: ChatLLM):
        self._llm = llm

    def run(
        self,
        session: Session,
        space: Space,
        trigger: str = "manual",
        max_sources: int = 20,
    ) -> ConsolidationRun:
        active = run_repo.find_active(session, space.id, stale_seconds=1800)
        if active is not None:
            return active

        pending = source_repo.list_pending(session, space.id, max_sources)
        run = run_repo.create(session, space.id, trigger, [s.id for s in pending])

        if not pending:
            return run_repo.finish(session, run, RunStatus.succeeded, atoms_touched=[])

        try:
            touched = self._consolidate(session, space, run, pending)
            return run_repo.finish(session, run, RunStatus.succeeded, atoms_touched=touched)
        except (LLMError, json.JSONDecodeError, KeyError, ValueError) as e:
            session.rollback()
            reason = f"LLMError[{e.code}]: {e}" if isinstance(e, LLMError) else f"{type(e).__name__}: {e}"
            return run_repo.finish(
                session, run, RunStatus.failed, error=reason
            )

    def _consolidate(
        self, session: Session, space: Space, run: ConsolidationRun, pending: list[Source]
    ) -> list[str]:
        active_atoms = atom_repo.list_active(session, space.id)
        index_text = prompts.render_index(active_atoms)
        sources_text = prompts.render_sources(pending)

        context_atoms: list[Atom] = []
        if active_atoms:
            plan = request_plan(self._llm,
                prompts.CONSOLIDATE_SELECT_SYSTEM,
                f"## atom 索引\n{index_text}\n\n## 最近经历\n{sources_text}",
                SelectPlan, "select_plan", run,
            )
            keys = plan.read
            by_key = {a.key: a for a in active_atoms}
            context_atoms = [by_key[k] for k in keys if k in by_key]

        today = date.today()
        plan = request_plan(self._llm,
            prompts.CONSOLIDATE_SYSTEM,
            f"## 当前时间\n{today.isoformat()}\n\n"
            f"## atom 索引\n{index_text}\n\n"
            f"## 相关原子全文\n{prompts.render_atoms_full(context_atoms)}\n\n"
            f"## 最近经历\n{sources_text}",
            WritePlan, "write_plan", run,
        )
        operations = [op.model_dump(mode="json", exclude_unset=True) for op in plan.operations]

        valid_ids = {s.id for s in pending}
        by_id = {s.id: s for s in pending}
        consumed: set[int] = set()
        touched: list[str] = []
        # 随本轮写入刷新，供后续 op 近重复合并
        working_atoms = list(active_atoms)
        for op in operations:
            op, _merge_note = apply_canonical_to_op(op, working_atoms)
            source_ids = [i for i in op.get("source_ids", []) if i in valid_ids]
            cited_text = "\n".join(
                by_id[i].content for i in source_ids if i in by_id
            )
            key = self._apply_op(
                session, space, run, op, source_ids, cited_text=cited_text, today=today
            )
            if key:
                touched.append(key)
                consumed.update(source_ids)
                working_atoms = atom_repo.list_active(session, space.id)

        # 未 cite：低优 skip；correction/高 salience 保持 pending（见 mark_consumed）
        source_repo.mark_consumed(session, pending, consumed)
        session.commit()
        return touched

    def _apply_op(
        self,
        session: Session,
        space: Space,
        run: ConsolidationRun,
        op: dict,
        source_ids: list[int],
        *,
        cited_text: str = "",
        today: date | None = None,
    ) -> str | None:
        kind = op.get("op")
        key = keyify(op.get("key") or "")
        if not key or kind not in ("upsert", "archive"):
            return None

        atom = atom_repo.get_by_key(session, space.id, key)
        reason = op.get("change_reason", "")

        if kind == "archive":
            if atom is None or atom.status == AtomStatus.archived:
                return None
            atom.status = AtomStatus.archived
            atom.updated_at = utcnow()
            self._record(session, atom, run, reason or "归档", source_ids)
            if atom.id is not None:
                embedding_repo.invalidate_for_atom(session, atom.id)
                self._invalidate_derived_downstream(session, atom.id)
            return key

        statement = (op.get("statement") or "").strip()[:_STATEMENT_MAX]
        detail = (op.get("detail") or "").strip()[:_DETAIL_MAX]
        if not statement:
            return None
        existing_on = atom.happened_on if atom is not None else None
        happened_on = resolve_happened_on(
            op.get("happened_on"),
            source_texts=cited_text,
            existing=existing_on,
            today=today,
        )
        detail = append_time_facts_to_detail(
            detail,
            happened_on,
            # A source can contain several unrelated events; derive time only from this fact.
            text=f"{statement}\n{detail}",
            today=today,
            source_anchor=source_time_anchor(cited_text),
            max_chars=_DETAIL_MAX,
        )
        confidence = op.get("confidence")
        atom_kind = AtomKind(op.get("kind") or AtomKind.belief.value)
        was_update = atom is not None
        # consolidate → L1；persona/self canonical → L3；op 可显式指定
        layer = default_layer_for_write(
            key=key,
            kind=atom_kind,
            explicit=op.get("memory_layer"),
        )

        if atom is None:
            atom = Atom(
                space_id=space.id,
                kind=atom_kind,
                key=key,
                statement=statement,
                detail=detail,
                happened_on=happened_on,
                confidence=confidence,
                memory_layer=layer,
            )
            session.add(atom)
            session.flush()
            reason = reason or "建原子"
        else:
            atom.kind = atom_kind
            atom.statement = statement
            atom.detail = detail
            atom.happened_on = happened_on
            if confidence is not None:
                atom.confidence = confidence
            # 不把已有 L2/L3 降回 L1（综合/画像产物）
            if atom.memory_layer not in (2, 3) or layer == 3:
                atom.memory_layer = layer
            atom.status = AtomStatus.active
            atom.updated_at = utcnow()
            reason = reason or "更新"

        self._record(session, atom, run, reason, source_ids)
        if atom.id is not None:
            embedding_repo.invalidate_for_atom(session, atom.id)
            if was_update:
                self._invalidate_derived_downstream(session, atom.id)
            # 仅当 op 显式带 links 时替换出边；缺省字段不改动已有边
            if "links" in op:
                self._apply_links(session, space.id, atom, op.get("links") or [])
        return key

    def _invalidate_derived_downstream(
        self, session: Session, atom_id: int
    ) -> None:
        from ..invalidation import invalidate_dependents
        invalidate_dependents(session, atom_id)

    def _apply_links(
        self,
        session: Session,
        space_id: int,
        atom: Atom,
        links: list,
    ) -> None:
        """解析 op.links；非法 kind / 不存在的 to key 静默丢弃。"""
        if atom.id is None:
            return
        resolved: list[tuple[int, AtomLinkKind, str | None]] = []
        for raw in links:
            if not isinstance(raw, dict):
                continue
            to_key = keyify(str(raw.get("to") or ""))
            kind_s = str(raw.get("kind") or "about")
            try:
                kind_e = AtomLinkKind(kind_s)
            except ValueError:
                continue
            if not to_key:
                continue
            peer = atom_repo.get_by_key(session, space_id, to_key)
            if peer is None or peer.id is None or peer.id == atom.id:
                continue
            note = raw.get("note")
            note_s = str(note) if note is not None else None
            resolved.append((peer.id, kind_e, note_s))
        atom_link_repo.replace_out_links(
            session,
            space_id=space_id,
            src_atom_id=atom.id,
            links=resolved,
        )

    def _record(
        self,
        session: Session,
        atom: Atom,
        run: ConsolidationRun,
        reason: str,
        source_ids: list[int],
    ) -> None:
        rev = revision_repo.add(
            session, atom, reason, RevisionTrigger.consolidation, run_id=run.id
        )
        evidence_repo.add_many(session, rev.id, source_ids)
