from datetime import date
from atom_memory.temporal import source_time_anchor, derive_time_facts


def test_relative_time_uses_source_timestamp_instead_of_event_date():
    anchor = source_time_anchor("[session_time] 1:56 pm on 8 May, 2023")
    assert anchor == date(2023, 5, 8)
    facts = derive_time_facts(date(2023, 5, 7), "went yesterday", source_anchor=anchor)
    assert not any("2023-05-06" in fact for fact in facts)
    assert "2023-05-07" in facts


def test_historical_year_and_month_are_grounded():
    anchor = date(2023, 5, 25)
    assert "2022" in derive_time_facts(None, "painted last year", source_anchor=anchor)
    assert "2023-06 (next month)" in derive_time_facts(None, "camping next month", source_anchor=anchor)
    assert derive_time_facts(None, "painted last year") == []


def test_multiple_different_anchors_are_ambiguous():
    assert source_time_anchor("[session_time] 2023-05-08\n[session_time] 2024-05-08") is None


def test_negated_relative_time_is_not_derived():
    assert derive_time_facts(None, "not last year and not next month", source_anchor=date(2023, 5, 8)) == []
