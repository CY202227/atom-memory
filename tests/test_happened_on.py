"""happened_on 解析：拒臆造年份、保留已有纠正日期。"""

from datetime import date

from atom_memory.consolidation.engine import resolve_happened_on


def test_reject_hallucinated_year():
    today = date(2026, 7, 27)
    assert (
        resolve_happened_on(
            "2024-07-27",
            source_texts="笔记：7月27日拔牙",
            existing=None,
            today=today,
        )
        is None
    )


def test_accept_current_year_partial_md():
    today = date(2026, 7, 27)
    assert resolve_happened_on(
        "2026-07-27",
        source_texts="今天拔牙了",
        existing=None,
        today=today,
    ) == date(2026, 7, 27)


def test_accept_year_explicit_in_source():
    today = date(2026, 7, 27)
    assert resolve_happened_on(
        "2025-03-01",
        source_texts="2025-03-01 入职",
        existing=None,
        today=today,
    ) == date(2025, 3, 1)


def test_null_preserves_existing():
    today = date(2026, 7, 27)
    existing = date(2026, 7, 22)
    assert (
        resolve_happened_on(
            None,
            source_texts="随便聊聊",
            existing=existing,
            today=today,
        )
        == existing
    )


def test_today_does_not_overwrite_grounded_correction():
    today = date(2026, 7, 27)
    existing = date(2026, 7, 22)
    assert (
        resolve_happened_on(
            "2026-07-27",
            source_texts="纠正：手术是 2026-07-22，不是今天",
            existing=existing,
            today=today,
        )
        == existing
    )


def test_explicit_today_can_update():
    today = date(2026, 7, 27)
    existing = date(2026, 7, 22)
    assert resolve_happened_on(
        "2026-07-27",
        source_texts="改一下：其实是今天做的手术",
        existing=existing,
        today=today,
    ) == date(2026, 7, 27)
