"""LoCoMo 形状评测管线：ingest → oracle 固化 → budgeted recall。

默认跑迷你 fixture（无外部依赖）。
官方集在 tests/fixtures/locomo10.json；完整跑分用::

    python -m atom_memory.eval.run_locomo --score qa
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from atom_memory.eval.locomo import (
    consolidate_sessions_oracle,
    evaluate_locomo_sample,
    ingest_locomo_sample,
    load_locomo,
    score_answer_in_text,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_FIXTURE = _FIXTURES / "locomo_mini.json"
_OFFICIAL = _FIXTURES / "locomo10.json"


def _uid(client) -> str:
    return client.post(
        "/spaces",
        json={"owner_id": "locomo", "subject_id": "bench"},
    ).json()["uid"]


def test_score_answer_in_text_normalizes():
    assert score_answer_in_text("On 7 May 2023 she went", "7 May 2023")
    assert score_answer_in_text("adopted Nori the tabby", "Nori")
    assert score_answer_in_text("year twenty twenty two", 2022) is False
    assert score_answer_in_text("painted in 2022", 2022)
    # 默认 coverage=1.0 保持硬逻辑
    assert score_answer_in_text(
        "pride parade school support",
        "pride parade school speech",
        coverage=1.0,
    ) is False



def test_locomo_mini_pipeline_under_budget(client, fake_llm):
    """迷你 LoCoMo：oracle 固化后，400 字预算召回应命中多数事实。"""
    samples = load_locomo(_FIXTURE)
    assert len(samples) == 1
    sample = samples[0]
    uid = _uid(client)

    source_ids = ingest_locomo_sample(client, uid, sample)
    assert len(source_ids) == 2

    consolidate_sessions_oracle(
        client, uid, sample, fake_llm, source_ids
    )

    atoms = client.get(f"/spaces/{uid}/atoms", params={"page_size": 50}).json()
    assert atoms["count"] >= 5

    report = evaluate_locomo_sample(
        client,
        uid,
        sample,
        method="bm25",
        max_atoms=5,
        budget_chars=400,
        detail="statement",
    )
    assert report.total == 5
    assert report.accuracy >= 0.8, (
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


def test_locomo_mini_evidence_recall(client, fake_llm):
    """迷你集：evidence recall@5 应接近满分（题少、evidence 单条）。"""
    sample = load_locomo(_FIXTURE)[0]
    uid = _uid(client)
    source_ids = ingest_locomo_sample(client, uid, sample)
    consolidate_sessions_oracle(client, uid, sample, fake_llm, source_ids)

    report = evaluate_locomo_sample(
        client,
        uid,
        sample,
        method="bm25",
        max_atoms=5,
        budget_chars=400,
        detail="statement",
        score_mode="evidence",
    )
    assert report.skipped_no_evidence == 0
    assert report.total == 5
    assert report.accuracy >= 0.8
    assert report.any_accuracy >= report.accuracy

    # 全量自检：evidence 应全部可召回
    all_report = evaluate_locomo_sample(
        client,
        uid,
        sample,
        method="all",
        max_atoms=50,
        budget_chars=20000,
        detail="full",
        score_mode="evidence",
    )
    assert all_report.accuracy == 1.0
    assert all_report.any_accuracy == 1.0


def test_locomo_budget_clips_when_tight(client, fake_llm):
    """极小预算时 chars_used 受限，且管线仍可返回报告。"""
    sample = load_locomo(_FIXTURE)[0]
    uid = _uid(client)
    source_ids = ingest_locomo_sample(client, uid, sample)
    consolidate_sessions_oracle(client, uid, sample, fake_llm, source_ids)

    report = evaluate_locomo_sample(
        client,
        uid,
        sample,
        max_atoms=8,
        budget_chars=80,
        detail="statement",
        max_questions=3,
    )
    assert report.total == 3
    assert report.avg_chars_used <= 80


@pytest.mark.locomo
def test_locomo_official_oracle_smoke(client, fake_llm):
    """可选：官方 locomo10.json 首条样本、前 2 session、前 10 题。

    默认读 tests/fixtures/locomo10.json；可用 ATOMMEM_LOCOMO_PATH 覆盖。
    仍用 oracle 固化（测召回/预算，不测真实 LLM 抽取）。
    """
    override = os.environ.get("ATOMMEM_LOCOMO_PATH", "").strip()
    p = Path(override) if override else _OFFICIAL
    if not p.is_file():
        pytest.skip(f"LoCoMo file not found: {p}")

    sample = load_locomo(p)[0]
    uid = _uid(client)
    source_ids = ingest_locomo_sample(
        client, uid, sample, max_sessions=2, salience=0.8
    )
    consolidate_sessions_oracle(
        client,
        uid,
        sample,
        fake_llm,
        source_ids,
        max_sessions=2,
    )
    report = evaluate_locomo_sample(
        client,
        uid,
        sample,
        budget_chars=400,
        max_atoms=5,
        detail="full",
        max_questions=10,
    )
    assert report.total == 10
    # 官方题含时序/对抗，oracle+短预算不保证高分；只要求管线跑通并有命中
    assert report.hits >= 1
    assert report.avg_chars_used <= 400
