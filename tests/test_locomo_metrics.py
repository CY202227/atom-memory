"""LoCoMo 评测：key 双射、evidence 计分、软覆盖率、oracle 日期富化。"""

from __future__ import annotations

from atom_memory.eval.locomo import (
    locomo_dia_key,
    oracle_write_plan,
    parse_session_date,
    score_answer_in_text,
    score_evidence_keys,
)
from atom_memory.linking import keyify


def test_locomo_dia_key_no_collision():
    k1 = locomo_dia_key("D1:11")
    k2 = locomo_dia_key("D11:1")
    assert k1 == "locomo-d1-11"
    assert k2 == "locomo-d11-1"
    assert k1 != k2
    # 旧写法会撞车
    assert keyify("locomo-D1:11") == keyify("locomo-D11:1")


def test_parse_session_date_and_oracle_enrich():
    dt = "1:56 pm on 8 May, 2023"
    d = parse_session_date(dt)
    assert d is not None
    assert d.isoformat() == "2023-05-08"
    assert d.strftime("%A") == "Monday"

    plan = oracle_write_plan(
        1,
        [
            {
                "dia_id": "D1:3",
                "speaker": "Caroline",
                "text": "I went to the LGBTQ support group yesterday.",
            },
            {
                "dia_id": "D11:1",
                "speaker": "Melanie",
                "text": "Different session turn that used to collide.",
            },
        ],
        date_time=dt,
    )
    ops = plan["operations"]
    assert len(ops) == 2
    assert ops[0]["key"] == "locomo-d1-3"
    assert ops[1]["key"] == "locomo-d11-1"
    assert ops[0]["happened_on"] == "2023-05-08"
    # 日期走 happened_on；statement 留给 speaker+正文，避免挤掉 BM25 内容词
    assert ops[0]["statement"].startswith("Caroline:")
    assert "2023-05-08" not in ops[0]["statement"]
    assert "weekday=Monday" in ops[0]["detail"]
    assert "session_time=" in ops[0]["detail"]
    assert "speaker=Caroline" in ops[0]["detail"]
    assert "time_facts=" in ops[0]["detail"]


def test_score_answer_coverage_soft_vs_hard():
    text = "pride parade school support group mentoring program"
    answer = "Pride parade, school speech, support group"
    # hard: missing "speech"
    assert score_answer_in_text(text, answer, coverage=1.0) is False
    # soft 0.6: most tokens present
    assert score_answer_in_text(text, answer, coverage=0.6) is True
    assert score_answer_in_text(text, answer, coverage=0.9) is False


def test_normalize_evidence_splits_compound():
    from atom_memory.eval.locomo import _normalize_evidence

    assert _normalize_evidence(["D8:6; D9:17"]) == ["D8:6", "D9:17"]
    assert _normalize_evidence(["D1:3", "D2:1"]) == ["D1:3", "D2:1"]
    assert _normalize_evidence("D1:5") == ["D1:5"]


def test_score_evidence_keys_any_all():
    gold = {"locomo-d1-3", "locomo-d2-1"}
    hit_all, hit_any = score_evidence_keys(
        {"locomo-d1-3", "locomo-d2-1", "x"}, gold
    )
    assert hit_all and hit_any
    hit_all, hit_any = score_evidence_keys({"locomo-d1-3"}, gold)
    assert not hit_all and hit_any
    hit_all, hit_any = score_evidence_keys({"other"}, gold)
    assert not hit_all and not hit_any
    hit_all, hit_any = score_evidence_keys(set(), set())
    assert not hit_all and not hit_any
