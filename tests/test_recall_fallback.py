"""inventory 保底召回与注入日期净化。"""

from datetime import date, datetime, timedelta, timezone

from atom_memory.models import Atom, AtomKind, AtomStatus
from atom_memory.recall.fallback import fallback_hits, is_inventory_query
from atom_memory.recall.base import RecallHit
from atom_memory.recall.render import (
    render_statement_block,
    should_prefix_happened_on,
)
from atom_memory.demo import memory_judge


def _atom(**kw) -> Atom:
    now = datetime.now(timezone.utc)
    defaults = dict(
        id=1,
        space_id=1,
        kind=AtomKind.person,
        key="k",
        statement="陈述",
        detail="",
        status=AtomStatus.active,
        schema_version=1,
        created_at=now,
        updated_at=now,
    )
    defaults.update(kw)
    return Atom(**defaults)


def test_is_inventory_query():
    assert is_inventory_query("你记得什么")
    assert is_inventory_query("what do you remember about me")
    assert not is_inventory_query("写日报要注意什么")


def test_fallback_hits_prefers_kinds_and_recency():
    older = _atom(
        id=1,
        key="old-belief",
        kind=AtomKind.belief,
        statement="旧认识",
        updated_at=datetime.now(timezone.utc) - timedelta(days=2),
    )
    newer = _atom(
        id=2,
        key="dental",
        kind=AtomKind.person,
        statement="牙齿敏感",
        updated_at=datetime.now(timezone.utc),
    )
    mid = _atom(
        id=3,
        key="lesson-1",
        kind=AtomKind.lesson,
        statement="教训一条",
        updated_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    hits = fallback_hits([older, newer, mid], exclude_keys=set(), limit=2)
    assert [h.atom.key for h in hits] == ["dental", "lesson-1"]


def test_should_prefix_skips_today_and_embedded_date():
    today = date(2026, 7, 27)
    assert not should_prefix_happened_on(today, "任意陈述", today=today)
    assert not should_prefix_happened_on(
        date(2026, 7, 22), "TA 于 7月22日拔牙", today=today
    )
    assert should_prefix_happened_on(
        date(2026, 7, 22), "TA 拔除下智齿", today=today
    )


def test_render_statement_block_omits_today_prefix():
    today = date(2026, 7, 27)
    atom = _atom(
        key="health",
        statement="TA 左上牙敏感",
        happened_on=today,
    )
    block = render_statement_block([RecallHit(atom=atom)], today=today)
    assert "2026-07-27" not in block
    assert "左上牙敏感" in block


def test_judge_system_encourages_preferences_and_one_shot():
    text = memory_judge._JUDGE_SYSTEM
    assert "即便只出现一次" in text
    assert "偏好" in text
    assert "纯机制" in text or "机制或科普" in text
