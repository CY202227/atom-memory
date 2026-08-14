"""PersonaMem 形状评测管线：ingest → oracle 固化 → budgeted recall。

默认跑迷你 fixture（无外部依赖）。完整跑分用::

    python -m atom_memory.eval.run_personamem --mini --score mcq
"""

from __future__ import annotations

import json
from pathlib import Path

from atom_memory.eval.personamem import (
    consolidate_sessions_oracle,
    evaluate_personamem_sample,
    ingest_personamem_sample,
    load_personamem,
    oracle_write_plan_personamem,
    parse_mcq_option,
)
from atom_memory.llm.base import ChatResult

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE = _FIXTURES / "personamem_mini.json"


def _uid(client) -> str:
    return client.post(
        "/spaces",
        json={"owner_id": "personamem", "subject_id": "bench"},
    ).json()["uid"]


class _ScriptedLLM:
    """按序返回预设文本；用于 mcq 离线测。"""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)

    def complete(
        self, system: str, user: str, response_format: dict | None = None
    ) -> ChatResult:
        del system, user, response_format
        if not self._answers:
            raise RuntimeError("ScriptedLLM: empty answers")
        return ChatResult(
            text=self._answers.pop(0), prompt_tokens=1, completion_tokens=1
        )


def test_parse_mcq_option():
    assert parse_mcq_option("B") == "B"
    assert parse_mcq_option("b)") == "B"
    assert parse_mcq_option("Answer: C") == "C"
    assert parse_mcq_option("I choose A because...") == "A"
    assert parse_mcq_option("") == ""


def test_oracle_write_plan_skips_assistant():
    plan = oracle_write_plan_personamem(
        1,
        [
            {"speaker": "User", "text": "I like green tea."},
            {"speaker": "Assistant", "text": "Noted."},
            {"speaker": "User", "text": "Call me Sam."},
        ],
        session_n=1,
    )
    ops = plan["operations"]
    assert len(ops) == 2
    assert all(op["kind"] == "belief" for op in ops)
    assert "green tea" in ops[0]["statement"]
    assert "Sam" in ops[1]["statement"]


def test_personamem_mini_pipeline_under_budget(client, fake_llm):
    """迷你 PersonaMem：oracle 固化后，400 字预算召回应命中多数偏好。"""
    samples = load_personamem(_FIXTURE)
    assert len(samples) == 2
    sample = samples[0]
    uid = _uid(client)

    source_ids = ingest_personamem_sample(client, uid, sample)
    assert len(source_ids) == 2

    consolidate_sessions_oracle(
        client, uid, sample, fake_llm, source_ids
    )

    atoms = client.get(
        f"/spaces/{uid}/atoms", params={"page_size": 50}
    ).json()
    assert atoms["count"] >= 3

    report = evaluate_personamem_sample(
        client,
        uid,
        sample,
        method="bm25",
        max_atoms=5,
        budget_chars=400,
        detail="statement",
        score_mode="retrieval",
    )
    assert report.total == 3
    assert report.accuracy >= 0.66, (
        f"accuracy={report.accuracy:.2f} hits={report.hits}/{report.total}; "
        + json.dumps(
            [
                {
                    "q": r.question,
                    "a": r.answer,
                    "hit": r.hit,
                    "chars": r.chars_used,
                    "ctx": r.context_block[:200],
                }
                for r in report.results
                if not r.hit
            ],
            ensure_ascii=False,
        )
    )
    assert report.avg_chars_used <= 400
    assert report.avg_chars_used > 0


def test_personamem_mini_mcq_with_scripted_llm(client, fake_llm):
    """离线 mcq：脚本 LLM 返回正确选项字母 → accuracy=1。"""
    sample = load_personamem(_FIXTURE)[0]
    uid = _uid(client)
    source_ids = ingest_personamem_sample(client, uid, sample)
    consolidate_sessions_oracle(client, uid, sample, fake_llm, source_ids)

    gold_letters = [q.correct_option for q in sample.qa]
    scripted = _ScriptedLLM(gold_letters)
    report = evaluate_personamem_sample(
        client,
        uid,
        sample,
        method="bm25",
        max_atoms=5,
        budget_chars=400,
        detail="statement",
        score_mode="mcq",
        llm=scripted,
    )
    assert report.total == 3
    assert report.accuracy == 1.0
    assert all(r.predicted_option == r.correct_option for r in report.results)


def test_personamem_budget_clips_when_tight(client, fake_llm):
    sample = load_personamem(_FIXTURE)[0]
    uid = _uid(client)
    source_ids = ingest_personamem_sample(client, uid, sample)
    consolidate_sessions_oracle(client, uid, sample, fake_llm, source_ids)

    report = evaluate_personamem_sample(
        client,
        uid,
        sample,
        max_atoms=8,
        budget_chars=80,
        detail="statement",
        max_questions=2,
        score_mode="retrieval",
    )
    assert report.total == 2
    assert report.avg_chars_used <= 80


def test_personamem_second_sample_loads(client, fake_llm):
    """第二个 persona 样本也能跑通管线。"""
    sample = load_personamem(_FIXTURE)[1]
    uid = _uid(client)
    source_ids = ingest_personamem_sample(client, uid, sample)
    consolidate_sessions_oracle(client, uid, sample, fake_llm, source_ids)
    report = evaluate_personamem_sample(
        client,
        uid,
        sample,
        budget_chars=400,
        max_atoms=5,
        score_mode="retrieval",
    )
    assert report.total == 2
    assert report.hits >= 1
