"""LoCoMo 形状数据的 ingest →（固化）→ recall 评测管线。

默认测检索/预算注入：答案是否出现在 `context_block`（含 statement）。
完整官方集请设 `ATOMMEM_LOCOMO_PATH`；CI 用 `tests/fixtures/locomo_mini.json`。

数据集：https://github.com/snap-research/locomo （CC BY-NC 4.0）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from atom_memory.linking import keyify
from atom_memory.llm.base import LLMError
from atom_memory.temporal import weekday_name

_CATEGORY_NAMES = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "open_domain",
    5: "adversarial",
}

_MONTH_ABBR = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# e.g. "1:56 pm on 8 May, 2023" / "8 May 2023"
_SESSION_DATE_RE = re.compile(
    r"(\d{1,2})\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+"
    r"(\d{4})",
    re.IGNORECASE,
)


class _HttpClient(Protocol):
    def post(self, url: str, json: dict | None = None, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class LocomoQAItem:
    question: str
    answer: Any
    evidence: list[str] = field(default_factory=list)
    category: int = 1


@dataclass(frozen=True)
class LocomoSample:
    sample_id: str
    conversation: dict[str, Any]
    qa: list[LocomoQAItem]


@dataclass
class LocomoQAResult:
    question: str
    answer: str
    category: int
    hit: bool
    chars_used: int
    atoms_clipped: int
    context_block: str
    prediction: str = ""
    f1: float = 0.0
    hit_any: bool = False
    gold_keys: list[str] = field(default_factory=list)
    recalled_keys: list[str] = field(default_factory=list)


@dataclass
class LocomoReport:
    sample_id: str
    total: int
    hits: int
    accuracy: float
    mean_f1: float
    avg_chars_used: float
    avg_atoms_clipped: float
    by_category: dict[str, dict[str, float]]
    results: list[LocomoQAResult]
    skipped_no_evidence: int = 0
    any_hits: int = 0
    any_accuracy: float = 0.0
    retrieval_coverage: float = 1.0


@dataclass
class LocomoBenchReport:
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
    sample_reports: list[LocomoReport]
    skipped_no_evidence: int = 0
    any_hits: int = 0
    any_accuracy: float = 0.0
    retrieval_coverage: float = 1.0

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
            "skipped_no_evidence": self.skipped_no_evidence,
            "any_hits": self.any_hits,
            "any_accuracy": round(self.any_accuracy, 4),
            "retrieval_coverage": self.retrieval_coverage,
            "by_category": {
                k: {
                    "total": int(v["total"]),
                    "hits": int(v["hits"]),
                    "accuracy": round(v["accuracy"], 4),
                    "mean_f1": round(v.get("mean_f1", 0.0), 4),
                    "any_hits": int(v.get("any_hits", 0)),
                    "any_accuracy": round(v.get("any_accuracy", 0.0), 4),
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
                    "skipped_no_evidence": s.skipped_no_evidence,
                    "any_hits": s.any_hits,
                    "any_accuracy": round(s.any_accuracy, 4),
                }
                for s in self.sample_reports
            ],
        }


def load_locomo(path: str | Path) -> list[LocomoSample]:
    """加载官方 `locomo10.json` 或同形状的迷你 fixture。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("LoCoMo root must be a JSON list")
    return [_parse_sample(item) for item in raw]


def _parse_sample(item: dict[str, Any]) -> LocomoSample:
    qa_raw = item.get("qa") or []
    qa = [
        LocomoQAItem(
            question=str(q["question"]),
            answer=q.get("answer", ""),
            evidence=_normalize_evidence(q.get("evidence") or []),
            category=int(q.get("category") or 1),
        )
        for q in qa_raw
    ]
    return LocomoSample(
        sample_id=str(item.get("sample_id") or "unknown"),
        conversation=dict(item.get("conversation") or {}),
        qa=qa,
    )


def _normalize_evidence(raw: Any) -> list[str]:
    """展开 evidence：支持 list，以及 ``D8:6; D9:17`` 合写串。"""
    items: list[Any]
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    out: list[str] = []
    for item in items:
        s = str(item or "").strip()
        if not s:
            continue
        parts = re.split(r"[;,]", s)
        for part in parts:
            dia = part.strip()
            if dia:
                out.append(dia)
    return out


def iter_sessions(
    conversation: dict[str, Any],
) -> list[tuple[int, str | None, list[dict[str, Any]]]]:
    """按 session 序号产出 (n, date_time, turns)。"""
    out: list[tuple[int, str | None, list[dict[str, Any]]]] = []
    for key, value in conversation.items():
        m = re.fullmatch(r"session_(\d+)", key)
        if not m or not isinstance(value, list):
            continue
        n = int(m.group(1))
        dt = conversation.get(f"session_{n}_date_time")
        dt_s = str(dt) if dt is not None else None
        out.append((n, dt_s, value))
    out.sort(key=lambda row: row[0])
    return out


def format_session_content(
    turns: list[dict[str, Any]],
    *,
    date_time: str | None = None,
) -> str:
    """把一轮 session 编成 source.content（含日期，便于时序题）。"""
    lines: list[str] = []
    if date_time:
        lines.append(f"[session_time] {date_time}")
    for turn in turns:
        speaker = str(turn.get("speaker") or "?")
        text = str(turn.get("text") or "").strip()
        dia = turn.get("dia_id")
        prefix = f"{speaker}"
        if dia:
            prefix = f"{speaker}({dia})"
        if text:
            lines.append(f"{prefix}：{text}")
    return "\n".join(lines)


def ingest_locomo_sample(
    client: _HttpClient,
    uid: str,
    sample: LocomoSample,
    *,
    salience: float = 0.7,
    max_sessions: int | None = None,
) -> list[int]:
    """每个 session 写成一条 `turn` source；返回 source id 列表。"""
    source_ids: list[int] = []
    sessions = iter_sessions(sample.conversation)
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    for n, date_time, turns in sessions:
        content = format_session_content(turns, date_time=date_time)
        if not content.strip():
            continue
        r = client.post(
            f"/spaces/{uid}/sources",
            json={
                "kind": "turn",
                "content": content,
                "salience": salience,
                "external_ref": {
                    "system": "locomo",
                    "sample_id": sample.sample_id,
                    "session": n,
                },
            },
        )
        if getattr(r, "status_code", 200) >= 400:
            raise RuntimeError(f"ingest session_{n} failed: {r.status_code} {r.text}")
        source_ids.append(int(r.json()["id"]))
    return source_ids


def locomo_dia_key(dia_id: str) -> str:
    """Map LoCoMo dia_id → atom key without collisions.

    ``D1:11`` and ``D11:1`` both used to collapse to ``locomo-d111`` via
    ``keyify("locomo-D1:11")``; keep the colon boundary as a hyphen.
    """
    raw = str(dia_id or "").strip()
    if not raw:
        return ""
    normalized = raw.lower().replace(":", "-")
    return keyify(f"locomo-{normalized}") or ""


def parse_session_date(date_time: str | None) -> date | None:
    """Parse LoCoMo session_*_date_time into a calendar date."""
    if not date_time:
        return None
    m = _SESSION_DATE_RE.search(str(date_time))
    if not m:
        return None
    day = int(m.group(1))
    month = _MONTH_ABBR.get(m.group(2)[:3].lower())
    year = int(m.group(3))
    if month is None:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def oracle_write_plan(
    source_id: int,
    turns: list[dict[str, Any]],
    *,
    date_time: str | None = None,
) -> dict[str, Any]:
    """无 LLM 时的固化计划：每条 turn → 一个 atom（测召回/预算用）。"""
    operations: list[dict[str, Any]] = []
    happened = parse_session_date(date_time)
    weekday = weekday_name(happened, lang="en") if happened else None
    date_prefix = f"{happened.isoformat()}|" if happened else ""

    for turn in turns:
        dia = str(turn.get("dia_id") or "").strip() or "turn"
        text = str(turn.get("text") or "").strip()
        if not text:
            continue
        speaker = str(turn.get("speaker") or "")
        if date_prefix:
            room = max(0, 80 - len(date_prefix))
            statement = (date_prefix + text[:room])[:80]
        else:
            statement = text[:80]

        meta_parts: list[str] = []
        if date_time:
            meta_parts.append(f"session_time={date_time}")
        if weekday:
            meta_parts.append(f"weekday={weekday}")
        if speaker:
            meta_parts.append(f"speaker={speaker}")
        meta = " | ".join(meta_parts)
        if meta:
            room = max(0, 300 - len(meta) - 3)
            detail = f"{text[:room]} | {meta}"[:300]
        else:
            detail = text[:300]

        key = locomo_dia_key(dia) or keyify(f"turn-{source_id}") or f"turn-{source_id}"
        op: dict[str, Any] = {
            "op": "upsert",
            "kind": "event",
            "key": key,
            "statement": statement,
            "detail": detail,
            "change_reason": "locomo-oracle",
            "source_ids": [source_id],
        }
        if happened is not None:
            op["happened_on"] = happened.isoformat()
        operations.append(op)
    return {"operations": operations}


def normalize_answer(answer: Any) -> str:
    if isinstance(answer, list):
        return " ".join(normalize_answer(a) for a in answer)
    return re.sub(r"\s+", " ", str(answer).strip().lower())


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^\w]+", normalize_answer(text)) if t]


def token_f1(prediction: str, answer: Any) -> float:
    """与常见 QA 评测一致的 token-level F1。"""
    pred_toks = _tokens(prediction)
    gold_toks = _tokens(str(answer) if not isinstance(answer, list) else normalize_answer(answer))
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common: dict[str, int] = {}
    for t in gold_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in pred_toks:
        n = common.get(t, 0)
        if n > 0:
            overlap += 1
            common[t] = n - 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def score_answer_in_text(
    text: str,
    answer: Any,
    *,
    coverage: float = 1.0,
) -> bool:
    """简易命中：标准答案（或分词片段）出现在召回文本中。

    coverage=1.0：全部 token（len>=2）须出现（历史硬逻辑）。
    coverage<1.0：命中 token 比例达到阈值即算命中（诊断口径）。
    """
    hay = normalize_answer(text)
    gold = normalize_answer(answer)
    if not gold:
        return False
    if gold in hay:
        return True
    tokens = [t for t in re.split(r"[^\w]+", gold) if len(t) >= 2]
    if not tokens:
        return gold in hay
    hit_n = sum(1 for t in tokens if t in hay)
    threshold = max(0.0, min(1.0, float(coverage)))
    if threshold >= 1.0:
        return hit_n == len(tokens)
    return (hit_n / len(tokens)) >= threshold


def score_evidence_keys(
    recalled_keys: set[str],
    gold_keys: set[str],
) -> tuple[bool, bool]:
    """Evidence recall：返回 (hit_all, hit_any)。"""
    if not gold_keys:
        return False, False
    hit_any = bool(recalled_keys & gold_keys)
    hit_all = gold_keys.issubset(recalled_keys)
    return hit_all, hit_any


_ANSWER_SYSTEM = (
    "Answer the question using ONLY the recalled memory below. "
    "Be concise. If unknown, reply exactly: unknown."
)


def answer_from_context(llm: Any, question: str, context_block: str) -> str:
    """用 LLM 基于召回块作答（论文式 LoCoMo QA）。"""
    user = (
        f"## Recalled memory\n{context_block or '(empty)'}\n\n"
        f"## Question\n{question}\n\n## Answer"
    )
    res = llm.complete(_ANSWER_SYSTEM, user)
    return (res.text or "").strip()


def evaluate_locomo_sample(
    client: _HttpClient,
    uid: str,
    sample: LocomoSample,
    *,
    method: str = "bm25",
    max_atoms: int = 5,
    budget_chars: int = 400,
    detail: str = "statement",
    categories: set[int] | None = None,
    max_questions: int | None = None,
    score_mode: str = "retrieval",
    llm: Any | None = None,
    hit_f1_threshold: float = 0.5,
    retrieval_coverage: float = 1.0,
) -> LocomoReport:
    """对 sample.qa 做 recall 并计分。

    score_mode:
      - retrieval: 金标是否出现在召回文本（测预算注入）
      - evidence: 金标 evidence dia_id → atom key 的 recall@k（主排序指标）
      - qa: LLM 据 context 作答，再算 token F1（更接近公开榜）
    """
    if score_mode not in ("retrieval", "qa", "evidence"):
        raise ValueError(f"unknown score_mode: {score_mode}")
    if score_mode == "qa" and llm is None:
        raise ValueError("score_mode=qa requires llm")

    qa_items = sample.qa
    if categories is not None:
        qa_items = [q for q in qa_items if q.category in categories]
    if max_questions is not None:
        qa_items = qa_items[:max_questions]

    results: list[LocomoQAResult] = []
    cat_hits: dict[int, list[bool]] = {}
    cat_any: dict[int, list[bool]] = {}
    cat_f1: dict[int, list[float]] = {}
    skipped_no_evidence = 0

    for item in qa_items:
        if score_mode == "evidence":
            gold_keys = {
                k
                for e in item.evidence
                if (k := locomo_dia_key(str(e)))
            }
            if not gold_keys:
                skipped_no_evidence += 1
                continue

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
        recalled_keys = [h.get("key") or "" for h in hits_out if h.get("key")]
        hit_text = context + "\n" + "\n".join(
            h.get("statement") or "" for h in hits_out
        )
        if detail == "full":
            hit_text += "\n" + "\n".join(
                h.get("detail") or "" for h in hits_out
            )

        prediction = ""
        gold_keys_list: list[str] = []
        hit_any = False
        if score_mode == "retrieval":
            ok = score_answer_in_text(
                hit_text, item.answer, coverage=retrieval_coverage
            )
            f1 = 1.0 if ok else 0.0
            prediction = "(retrieval)"
            hit_any = ok
        elif score_mode == "evidence":
            gold_keys_list = sorted(gold_keys)
            ok, hit_any = score_evidence_keys(set(recalled_keys), gold_keys)
            f1 = 1.0 if ok else 0.0
            prediction = (
                f"(evidence all={ok} any={hit_any} "
                f"gold={len(gold_keys)} hit={len(set(recalled_keys) & gold_keys)})"
            )
        else:
            try:
                prediction = answer_from_context(llm, item.question, context)
            except LLMError as e:
                # 单题超时/失败不中断整场；计入未命中
                prediction = f"error: {e}"
            f1 = token_f1(prediction, item.answer)
            ok = f1 >= hit_f1_threshold or score_answer_in_text(
                prediction, item.answer, coverage=retrieval_coverage
            )
            hit_any = ok

        results.append(
            LocomoQAResult(
                question=item.question,
                answer=normalize_answer(item.answer),
                category=item.category,
                hit=ok,
                chars_used=int(body.get("chars_used") or 0),
                atoms_clipped=int(body.get("atoms_clipped") or 0),
                context_block=context,
                prediction=prediction,
                f1=f1,
                hit_any=hit_any,
                gold_keys=gold_keys_list,
                recalled_keys=recalled_keys,
            )
        )
        cat_hits.setdefault(item.category, []).append(ok)
        cat_any.setdefault(item.category, []).append(hit_any)
        cat_f1.setdefault(item.category, []).append(f1)

    total = len(results)
    hits = sum(1 for x in results if x.hit)
    any_hits = sum(1 for x in results if x.hit_any)
    mean_f1 = (sum(x.f1 for x in results) / total) if total else 0.0
    by_category: dict[str, dict[str, float]] = {}
    for cat, flags in sorted(cat_hits.items()):
        name = _CATEGORY_NAMES.get(cat, str(cat))
        f1s = cat_f1.get(cat) or []
        anys = cat_any.get(cat) or []
        by_category[name] = {
            "total": float(len(flags)),
            "hits": float(sum(flags)),
            "accuracy": (sum(flags) / len(flags)) if flags else 0.0,
            "mean_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
            "any_hits": float(sum(anys)),
            "any_accuracy": (sum(anys) / len(anys)) if anys else 0.0,
        }

    return LocomoReport(
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
        skipped_no_evidence=skipped_no_evidence,
        any_hits=any_hits,
        any_accuracy=(any_hits / total) if total else 0.0,
        retrieval_coverage=float(retrieval_coverage),
    )


def aggregate_reports(
    reports: list[LocomoReport],
    *,
    mode: str,
) -> LocomoBenchReport:
    total = sum(r.total for r in reports)
    hits = sum(r.hits for r in reports)
    any_hits = sum(r.any_hits for r in reports)
    skipped = sum(r.skipped_no_evidence for r in reports)
    f1_sum = sum(r.mean_f1 * r.total for r in reports)
    chars_sum = sum(r.avg_chars_used * r.total for r in reports)
    clip_sum = sum(r.avg_atoms_clipped * r.total for r in reports)
    coverage = reports[0].retrieval_coverage if reports else 1.0
    cat_hits: dict[str, list[bool]] = {}
    cat_any: dict[str, list[bool]] = {}
    cat_f1: dict[str, list[float]] = {}
    for report in reports:
        for result in report.results:
            name = _CATEGORY_NAMES.get(result.category, str(result.category))
            cat_hits.setdefault(name, []).append(result.hit)
            cat_any.setdefault(name, []).append(result.hit_any)
            cat_f1.setdefault(name, []).append(result.f1)
    by_category: dict[str, dict[str, float]] = {}
    for name, flags in sorted(cat_hits.items()):
        f1s = cat_f1.get(name) or []
        anys = cat_any.get(name) or []
        by_category[name] = {
            "total": float(len(flags)),
            "hits": float(sum(flags)),
            "accuracy": (sum(flags) / len(flags)) if flags else 0.0,
            "mean_f1": (sum(f1s) / len(f1s)) if f1s else 0.0,
            "any_hits": float(sum(anys)),
            "any_accuracy": (sum(anys) / len(anys)) if anys else 0.0,
        }
    return LocomoBenchReport(
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
        skipped_no_evidence=skipped,
        any_hits=any_hits,
        any_accuracy=(any_hits / total) if total else 0.0,
        retrieval_coverage=coverage,
    )


def format_scoreboard(bench: LocomoBenchReport) -> str:
    lines = [
        "========== LoCoMo Score ==========",
        f"mode:              {bench.mode}",
        f"samples:           {bench.samples}",
        f"questions:         {bench.total}",
        f"accuracy:          {bench.accuracy:.4f}  ({bench.hits}/{bench.total})",
        f"mean_f1:           {bench.mean_f1:.4f}",
        f"avg_chars_used:    {bench.avg_chars_used:.1f}",
        f"avg_atoms_clipped: {bench.avg_atoms_clipped:.2f}",
    ]
    if "score=evidence" in bench.mode:
        lines.append(
            f"any_accuracy:      {bench.any_accuracy:.4f}  "
            f"({bench.any_hits}/{bench.total})"
        )
        lines.append(f"skipped_no_ev:     {bench.skipped_no_evidence}")
    if "score=retrieval" in bench.mode:
        lines.append(f"retrieval_coverage:{bench.retrieval_coverage:g}")
    lines.append("---------- by category ----------")
    for name, stats in bench.by_category.items():
        extra = ""
        if "score=evidence" in bench.mode:
            extra = (
                f"  any={stats.get('any_accuracy', 0.0):.4f}"
                f"({int(stats.get('any_hits', 0))}/{int(stats['total'])})"
            )
        lines.append(
            f"  {name:14s}  acc={stats['accuracy']:.4f}  "
            f"f1={stats.get('mean_f1', 0.0):.4f}  "
            f"({int(stats['hits'])}/{int(stats['total'])}){extra}"
        )
    lines.append("==================================")
    return "\n".join(lines)


def consolidate_sessions_oracle(
    client: _HttpClient,
    uid: str,
    sample: LocomoSample,
    fake_llm: Any,
    source_ids: list[int],
    *,
    max_sessions: int | None = None,
) -> None:
    """用 oracle_write_plan + FakeLLM 逐 session 固化（无真实抽取模型）。"""
    sessions = iter_sessions(sample.conversation)
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
    if len(source_ids) != len(sessions):
        raise ValueError(
            f"source_ids ({len(source_ids)}) != sessions ({len(sessions)})"
        )
    for source_id, (_n, date_time, turns) in zip(source_ids, sessions):
        plan = oracle_write_plan(source_id, turns, date_time=date_time)
        # 库非空后 consolidate 会先 select
        atoms = client.get(f"/spaces/{uid}/atoms", params={"page_size": 1}).json()
        has_atoms = int(atoms.get("count") or 0) > 0
        if has_atoms:
            fake_llm.responses = [
                json.dumps({"read": []}),
                json.dumps(plan, ensure_ascii=False),
            ]
        else:
            fake_llm.responses = [json.dumps(plan, ensure_ascii=False)]
        r = client.post(f"/spaces/{uid}/consolidate", json={"max_sources": 1})
        if getattr(r, "status_code", 200) >= 400:
            raise RuntimeError(f"consolidate failed: {r.status_code} {r.text}")
        body = r.json()
        if body.get("status") != "succeeded":
            raise RuntimeError(f"consolidate not succeeded: {body}")
