"""固化引擎：Atom-first，wiki 唯一的常规写通路。

两阶段：select（选 key 读 detail）→ write（产出 upsert/archive）。
"""

import json
import re
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel
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
from ..repositories import (
    atom_repo,
    evidence_repo,
    revision_repo,
    run_repo,
    source_repo,
)
from . import prompts
from .canonical import apply_canonical_to_op

_STATEMENT_MAX = 80
_DETAIL_MAX = 300


def parse_json_object(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


class AtomOp(BaseModel):
    op: Literal["upsert", "archive"]
    kind: AtomKind = AtomKind.belief
    key: str
    statement: str = ""
    detail: str = ""
    happened_on: Optional[str] = None
    confidence: Optional[float] = None
    change_reason: str = ""
    source_ids: list[int] = []


class WritePlan(BaseModel):
    operations: list[AtomOp]


class SelectPlan(BaseModel):
    read: list[str]


def _json_schema_format(name: str, model: type[BaseModel]) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": model.model_json_schema()},
    }


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
            return run_repo.finish(
                session, run, RunStatus.failed, error=f"{type(e).__name__}: {e}"
            )

    def _consolidate(
        self, session: Session, space: Space, run: ConsolidationRun, pending: list[Source]
    ) -> list[str]:
        active_atoms = atom_repo.list_active(session, space.id)
        index_text = prompts.render_index(active_atoms)
        sources_text = prompts.render_sources(pending)

        context_atoms: list[Atom] = []
        if active_atoms:
            res = self._llm.complete(
                prompts.CONSOLIDATE_SELECT_SYSTEM,
                f"## atom 索引\n{index_text}\n\n## 最近经历\n{sources_text}",
                response_format=_json_schema_format("select_plan", SelectPlan),
            )
            run.prompt_tokens += res.prompt_tokens
            run.completion_tokens += res.completion_tokens
            keys = parse_json_object(res.text).get("read", [])
            by_key = {a.key: a for a in active_atoms}
            context_atoms = [by_key[k] for k in keys if k in by_key]

        today = date.today()
        res = self._llm.complete(
            prompts.CONSOLIDATE_SYSTEM,
            f"## 当前时间\n{today.isoformat()}\n\n"
            f"## atom 索引\n{index_text}\n\n"
            f"## 相关原子全文\n{prompts.render_atoms_full(context_atoms)}\n\n"
            f"## 最近经历\n{sources_text}",
            response_format=_json_schema_format("write_plan", WritePlan),
        )
        run.prompt_tokens += res.prompt_tokens
        run.completion_tokens += res.completion_tokens
        operations = parse_json_object(res.text).get("operations", [])

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
        confidence = op.get("confidence")
        atom_kind = AtomKind(op.get("kind") or AtomKind.belief.value)

        if atom is None:
            atom = Atom(
                space_id=space.id,
                kind=atom_kind,
                key=key,
                statement=statement,
                detail=detail,
                happened_on=happened_on,
                confidence=confidence,
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
            atom.status = AtomStatus.active
            atom.updated_at = utcnow()
            reason = reason or "更新"

        self._record(session, atom, run, reason, source_ids)
        return key

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
