"""PersonaMem 形状数据的 ingest →（固化）→ recall 评测管线。

测用户画像 / 隐式偏好在预算约束下的召回与个性化作答。
CI 用 `tests/fixtures/personamem_mini.json`；完整集可经
`ATOMMEM_PERSONAMEM_PATH` 或 CLI 路径传入。

参考：https://github.com/bowen-upenn/PersonaMem （arXiv:2504.14225）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from atom_memory.eval.locomo import (
    answer_from_context,
    normalize_answer,
    score_answer_in_text,
    token_f1,
)
from atom_memory.linking import keyify
from atom_memory.llm.base import LLMError


class _HttpClient(Protocol):
    def post(self, url: str, json: dict | None = None, **kwargs: Any) -> Any: ...

    def get(self, url: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class PersonaMemQAItem:
    question: str
    answer: Any
    options: list[str] = field(default_factory=list)
    correct_option: str = ""
    category: str = "preference"


@dataclass(frozen=True)
class PersonaMemSample:
    sample_id: str
    persona_id: str
    conversation: dict[str, Any]
    qa: list[PersonaMemQAItem]


@dataclass
class PersonaMemQAResult:
    question: str
    answer: str
    category: str
    hit: bool
    chars_used: int
    atoms_clipped: int
    context_block: str
    prediction: str = ""
    f1: float = 0.0
    correct_option: str = ""
    predicted_option: str = ""


@dataclass
class PersonaMemReport:
    sample_id: str
    total: int
    hits: int
    accuracy: float
    mean_f1: float
    avg_chars_used: float
    avg_atoms_clipped: float
    by_category: dict[str, dict[str, float]]
    results: list[PersonaMemQAResult]


@dataclass
class PersonaMemBenchReport:
    """多 sample 汇总得分。"""

    mode: str
    samples: int
    total: int
    hits: int
    accuracy: float
    mean_f1: float
    avg_chars_used: float
    avg_atoms_clipped: float
    by_category: dict[str, dict[str, float]]
    sample_reports: list[PersonaMemReport]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "samples": self.samples,
            "total": self.total,
            "hits": self.hits,
            "accuracy": round(self.accuracy, 4),
            "mean_f1": round(self.mean_f1, 4),
            "avg_chars_used": round(self.avg_chars_used, 1),
            "avg_atoms_clipped": round(self.avg_atoms_clipped, 2),
            "by_category": {
                k: {
                    "total": int(v["total"]),
                    "hits": int(v["hits"]),
                    "accuracy": round(v["accuracy"], 4),
                    "mean_f1": round(v.get("mean_f1", 0.0), 4),
                }
                for k, v in self.by_category.items()
            },
            "samples_detail": [
                {
                    "sample_id": s.sample_id,
                    "total": s.total,
                    "hits": s.hits,
                    "accuracy": round(s.accuracy, 4),
                    "mean_f1": round(s.mean_f1, 4),
                    "avg_chars_used": round(s.avg_chars_used, 1),
                }
                for s in self.sample_reports
            ],
        }


def load_personamem(path: str | Path) -> list[PersonaMemSample]:
    """加载 PersonaMem 形状 JSON（list of samples）。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("PersonaMem root must be a JSON list")
    return [_parse_sample(item) for item in raw]


def _parse_sample(item: dict[str, Any]) -> PersonaMemSample:
    qa_raw = item.get("qa") or []
    qa = [
        PersonaMemQAItem(
            question=str(q["question"]),
            answer=q.get("answer", ""),
            options=[str(o) for o in (q.get("options") or [])],
            correct_option=str(q.get("correct_option") or "").strip().upper(),
            category=str(q.get("category") or "preference"),
        )
        for q in qa_raw
    ]
    return PersonaMemSample(
        sample_id=str(item.get("sample_id") or "unknown"),
        persona_id=str(item.get("persona_id") or item.get("sample_id") or ""),
        conversation=dict(item.get("conversation") or {}),
        qa=qa,
    )


def iter_sessions(
    conversation: dict[str, Any],
) -> list[tuple[int, list[dict[str, Any]]]]:
    """按 session 序号产出 (n, turns)。"""
    out: list[tuple[int, list[dict[str, Any]]]] = []
    for key, value in conversation.items():
        m = re.fullmatch(r"session_(\d+)", key)
        if not m or not isinstance(value, list):
            continue
        out.append((int(m.group(1)), value))
    out.sort(key=lambda row: row[0])
    return out


def format_session_content(turns: list[dict[str, Any]]) -> str:
    """把一轮 session 编成 source.content。"""
    lines: list[str] = []
    for turn in turns:
        speaker = str(turn.get("speaker") or "?")
        text = str(turn.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}：{text}")
    return "\n".join(lines)


def ingest_personamem_sample(
    client: _HttpClient,
    uid: str,
    sample: PersonaMemSample,
    *,
    salience: float = 0.8,
    max_sessions: int | None = None,
) -> list[int]:
    """每个 session 写成一条 `turn` source；返回 source id 列表。"""
    source_ids: list[int] = []
    sessions = iter_sessions(sample.conversation)
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    for n, turns in sessions:
        content = format_session_content(turns)
        if not content.strip():
            continue
        r = client.post(
            f"/spaces/{uid}/sources",
            json={
                "kind": "turn",
                "content": content,
                "salience": salience,
                "external_ref": {
                    "system": "personamem",
                    "sample_id": sample.sample_id,
                    "persona_id": sample.persona_id,
                    "session": n,
                },
            },
        )
        if getattr(r, "status_code", 200) >= 400:
            raise RuntimeError(
                f"ingest session_{n} failed: {r.status_code} {r.text}"
            )
        source_ids.append(int(r.json()["id"]))
    return source_ids


def oracle_write_plan_personamem(
    source_id: int,
    turns: list[dict[str, Any]],
    *,
    session_n: int = 1,
) -> dict[str, Any]:
    """无 LLM 时的固化计划：用户 turn → preference/persona atom。"""
    operations: list[dict[str, Any]] = []
    user_idx = 0
    for turn in turns:
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        speaker = str(turn.get("speaker") or "").strip().lower()
        # 助理句通常不作为偏好事实入库
        if speaker in ("assistant", "ai", "bot", "system"):
            continue
        user_idx += 1
        kind = "belief"
        key_base = f"pm-s{session_n}-u{user_idx}"
        key = keyify(key_base) or key_base
        statement = text[:80]
        detail = text[:300]
        operations.append(
            {
                "op": "upsert",
                "kind": kind,
                "key": key,
                "statement": statement,
                "detail": detail,
                "change_reason": "personamem-oracle",
                "source_ids": [source_id],
            }
        )
    return {"operations": operations}


def consolidate_sessions_oracle(
    client: _HttpClient,
    uid: str,
    sample: PersonaMemSample,
    fake_llm: Any,
    source_ids: list[int],
    *,
    max_sessions: int | None = None,
) -> None:
    """用 oracle_write_plan_personamem + FakeLLM 逐 session 固化。"""
    sessions = iter_sessions(sample.conversation)
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    if len(source_ids) != len(sessions):
        raise ValueError(
            f"source_ids ({len(source_ids)}) != sessions ({len(sessions)})"
        )
    for source_id, (n, turns) in zip(source_ids, sessions):
        plan = oracle_write_plan_personamem(
            source_id, turns, session_n=n
        )
        atoms = client.get(
            f"/spaces/{uid}/atoms", params={"page_size": 1}
        ).json()
        has_atoms = int(atoms.get("count") or 0) > 0
        if has_atoms:
            fake_llm.responses = [
                json.dumps({"read": []}),
                json.dumps(plan, ensure_ascii=False),
            ]
        else:
            fake_llm.responses = [json.dumps(plan, ensure_ascii=False)]
        r = client.post(
            f"/spaces/{uid}/consolidate", json={"max_sources": 1}
        )
        if getattr(r, "status_code", 200) >= 400:
            raise RuntimeError(
                f"consolidate failed: {r.status_code} {r.text}"
            )
        body = r.json()
        if body.get("status") != "succeeded":
            raise RuntimeError(f"consolidate not succeeded: {body}")


_MCQ_SYSTEM = (
    "Answer the multiple-choice question using ONLY the recalled memory. "
    "Reply with a single letter (A, B, C, or D). No explanation."
)


def parse_mcq_option(text: str) -> str:
    """从模型输出中提取选项字母。"""
    raw = (text or "").strip().upper()
    if not raw:
        return ""
    # 优先行首字母
    m = re.match(r"^([A-D])\b", raw)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b", raw)
    if m:
        return m.group(1)
    m = re.search(r"([A-D])\)", raw)
    if m:
        return m.group(1)
    return ""


def answer_mcq_from_context(
    llm: Any,
    question: str,
    options: list[str],
    context_block: str,
) -> str:
    """用 LLM 基于召回块做选择题。"""
    opts = "\n".join(options) if options else "(no options)"
    user = (
        f"## Recalled memory\n{context_block or '(empty)'}\n\n"
        f"## Question\n{question}\n\n"
        f"## Options\n{opts}\n\n"
        f"## Answer (letter only)"
    )
    res = llm.complete(_MCQ_SYSTEM, user)
    return (res.text or "").strip()


def evaluate_personamem_sample(
    client: _HttpClient,
    uid: str,
    sample: PersonaMemSample,
    *,
    method: str = "bm25",
    max_atoms: int = 5,
    budget_chars: int = 400,
    detail: str = "statement",
    max_questions: int | None = None,
    score_mode: str = "retrieval",
    llm: Any | None = None,
    hit_f1_threshold: float = 0.5,
    retrieval_coverage: float = 1.0,
) -> PersonaMemReport:
    """对 sample.qa 做 recall 并计分。

    score_mode:
      - retrieval: 金标是否出现在召回文本
      - mcq: LLM 选 A/B/C/D，与 correct_option 比对
      - qa: LLM 自由作答，再算 token F1
    """
    if score_mode not in ("retrieval", "mcq", "qa"):
        raise ValueError(f"unknown score_mode: {score_mode}")
    if score_mode in ("mcq", "qa") and llm is None:
        raise ValueError(f"score_mode={score_mode} requires llm")

    qa_items = sample.qa
    if max_questions is not None:
        qa_items = qa_items[:max_questions]

    results: list[PersonaMemQAResult] = []
    cat_hits: dict[str, list[bool]] = {}
    cat_f1: dict[str, list[float]] = {}

    for item in qa_items:
        r = client.post(
            f"/spaces/{uid}/recall",
            json={
                "query": item.question,
                "method": method,
                "max_atoms": max_atoms,
                "budget_chars": budget_chars,
                "include_recent_sources": False,
                "detail": detail,
            },
        )
        if getattr(r, "status_code", 200) >= 400:
            raise RuntimeError(f"recall failed: {r.status_code} {r.text}")
        body = r.json()
        context = body.get("context_block") or ""
        hits_out = body.get("hits") or []
        hit_text = context + "\n" + "\n".join(
            h.get("statement") or "" for h in hits_out
        )
        if detail == "full":
            hit_text += "\n" + "\n".join(
                h.get("detail") or "" for h in hits_out
            )

        prediction = ""
        predicted_option = ""
        if score_mode == "retrieval":
            ok = score_answer_in_text(
                hit_text, item.answer, coverage=retrieval_coverage
            )
            f1 = 1.0 if ok else 0.0
            prediction = "(retrieval)"
        elif score_mode == "mcq":
            try:
                prediction = answer_mcq_from_context(
                    llm, item.question, item.options, context
                )
            except LLMError as e:
                prediction = f"error: {e}"
            predicted_option = parse_mcq_option(prediction)
            gold = (item.correct_option or "").upper()
            ok = bool(gold) and predicted_option == gold
            f1 = 1.0 if ok else 0.0
        else:
            try:
                prediction = answer_from_context(
                    llm, item.question, context
                )
            except LLMError as e:
                prediction = f"error: {e}"
            f1 = token_f1(prediction, item.answer)
            ok = f1 >= hit_f1_threshold or score_answer_in_text(
                prediction, item.answer, coverage=retrieval_coverage
            )

        results.append(
            PersonaMemQAResult(
                question=item.question,
                answer=normalize_answer(item.answer),
                category=item.category,
                hit=ok,
                chars_used=int(body.get("chars_used") or 0),
                atoms_clipped=int(body.get("atoms_clipped") or 0),
                context_block=context,
                prediction=prediction,
                f1=f1,
                correct_option=item.correct_option,
                predicted_option=predicted_option,
            )
        )
        cat_hits.setdefault(item.category, []).append(ok)
        cat_f1.setdefault(item.category, []).append(f1)

    total = len(results)
    hits = sum(1 for x in results if x.hit)
    mean_f1 = (sum(x.f1 for x in results) / total) if total else 0.0
    by_category: dict[str, dict[str, float]] = {}
    for cat, flags in sorted(cat_hits.items()):
        f1s = cat_f1.get(cat) or []
        by_category[cat] = {
            "total": float(len(flags)),
            "hits": float(sum(flags)),
            "accuracy": (sum(flags) / len(flags)) if flags else 0.0,
            "mean_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
        }

    return PersonaMemReport(
        sample_id=sample.sample_id,
        total=total,
        hits=hits,
        accuracy=(hits / total) if total else 0.0,
        mean_f1=mean_f1,
        avg_chars_used=(
            sum(x.chars_used for x in results) / total if total else 0.0
        ),
        avg_atoms_clipped=(
            sum(x.atoms_clipped for x in results) / total if total else 0.0
        ),
        by_category=by_category,
        results=results,
    )


def aggregate_reports(
    reports: list[PersonaMemReport],
    *,
    mode: str,
) -> PersonaMemBenchReport:
    total = sum(r.total for r in reports)
    hits = sum(r.hits for r in reports)
    f1_sum = sum(r.mean_f1 * r.total for r in reports)
    chars_sum = sum(r.avg_chars_used * r.total for r in reports)
    clip_sum = sum(r.avg_atoms_clipped * r.total for r in reports)
    cat_hits: dict[str, list[bool]] = {}
    cat_f1: dict[str, list[float]] = {}
    for report in reports:
        for result in report.results:
            cat_hits.setdefault(result.category, []).append(result.hit)
            cat_f1.setdefault(result.category, []).append(result.f1)
    by_category: dict[str, dict[str, float]] = {}
    for name, flags in sorted(cat_hits.items()):
        f1s = cat_f1.get(name) or []
        by_category[name] = {
            "total": float(len(flags)),
            "hits": float(sum(flags)),
            "accuracy": (sum(flags) / len(flags)) if flags else 0.0,
            "mean_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
        }
    return PersonaMemBenchReport(
        mode=mode,
        samples=len(reports),
        total=total,
        hits=hits,
        accuracy=(hits / total) if total else 0.0,
        mean_f1=(f1_sum / total) if total else 0.0,
        avg_chars_used=(chars_sum / total) if total else 0.0,
        avg_atoms_clipped=(clip_sum / total) if total else 0.0,
        by_category=by_category,
        sample_reports=reports,
    )


def format_personamem_scoreboard(bench: PersonaMemBenchReport) -> str:
    lines = [
        "========== PersonaMem Score ==========",
        f"mode:              {bench.mode}",
        f"samples:           {bench.samples}",
        f"questions:         {bench.total}",
        f"accuracy:          {bench.accuracy:.4f}  ({bench.hits}/{bench.total})",
        f"mean_f1:           {bench.mean_f1:.4f}",
        f"avg_chars_used:    {bench.avg_chars_used:.1f}",
        f"avg_atoms_clipped: {bench.avg_atoms_clipped:.2f}",
        "---------- by category ----------",
    ]
    for name, stats in bench.by_category.items():
        lines.append(
            f"  {name:14s}  acc={stats['accuracy']:.4f}  "
            f"f1={stats.get('mean_f1', 0.0):.4f}  "
            f"({int(stats['hits'])}/{int(stats['total'])})"
        )
    lines.append("======================================")
    return "\n".join(lines)
