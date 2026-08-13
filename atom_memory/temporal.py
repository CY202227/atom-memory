"""纯函数时间派生：星期几、相对日期解析、可注入 detail 的派生表述。"""

from __future__ import annotations

import re
from datetime import date, timedelta

_WEEKDAY_ZH = (
    "星期一",
    "星期二",
    "星期三",
    "星期四",
    "星期五",
    "星期六",
    "星期日",
)

_WEEKDAY_EN = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

_WEEKDAY_EN_TO_I = {name.lower(): i for i, name in enumerate(_WEEKDAY_EN)}
_WEEKDAY_ZH_TO_I = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
}

# 上周四 / 上周一
_REL_LAST_WEEKDAY_ZH = re.compile(
    r"上(?:个)?周([一二三四五六日天])"
)
# 前一个周日 / 前一个 Sunday
_REL_PREV_WEEKDAY = re.compile(
    r"(?:the\s+)?(?:sunday|monday|tuesday|wednesday|thursday|friday|saturday)"
    r"\s+before|"
    r"前一个?(?:星期|周)?([一二三四五六日天])|"
    r"前一个?\s*(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)",
    re.IGNORECASE,
)
_REL_BEFORE_DATE = re.compile(
    r"(?:the\s+)?"
    r"(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)"
    r"\s+before\s+"
    r"(\d{1,2})\s+"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s*"
    r"(\d{4})",
    re.IGNORECASE,
)

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


def weekday_name(d: date, *, lang: str = "zh") -> str:
    """Return weekday label for a date (zh: 星期一 / en: Monday)."""
    i = d.weekday()  # Mon=0
    if lang == "en":
        return _WEEKDAY_EN[i]
    return _WEEKDAY_ZH[i]


def _prev_weekday(anchor: date, target_weekday: int) -> date:
    """Strictly before anchor, the most recent date with target weekday (Mon=0)."""
    delta = (anchor.weekday() - target_weekday) % 7
    if delta == 0:
        delta = 7
    return anchor - timedelta(days=delta)


def resolve_relative(expr: str, anchor: date) -> date | None:
    """解析相对日期表述；无法解析返回 None。"""
    text = (expr or "").strip()
    if not text:
        return None

    m = _REL_BEFORE_DATE.search(text)
    if m:
        wd = _WEEKDAY_EN_TO_I.get(m.group(1).lower())
        month = _MONTH_ABBR.get(m.group(3)[:3].lower())
        if wd is None or month is None:
            return None
        try:
            ref = date(int(m.group(4)), month, int(m.group(2)))
        except ValueError:
            return None
        return _prev_weekday(ref, wd)

    m = _REL_LAST_WEEKDAY_ZH.search(text)
    if m:
        wd = _WEEKDAY_ZH_TO_I.get(m.group(1))
        if wd is None:
            return None
        # 「上周X」：上一整周的那一天
        # 先退到本周一，再退 7 天到上周一，再加 weekday
        this_monday = anchor - timedelta(days=anchor.weekday())
        last_monday = this_monday - timedelta(days=7)
        return last_monday + timedelta(days=wd)

    m = re.search(
        r"前一个?\s*(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)",
        text,
        re.IGNORECASE,
    )
    if m:
        wd = _WEEKDAY_EN_TO_I.get(m.group(1).lower())
        if wd is not None:
            return _prev_weekday(anchor, wd)

    m = re.search(r"前一个?(?:星期|周)?([一二三四五六日天])", text)
    if m:
        wd = _WEEKDAY_ZH_TO_I.get(m.group(1))
        if wd is not None:
            return _prev_weekday(anchor, wd)

    if "昨天" in text or "昨日" in text:
        return anchor - timedelta(days=1)
    if "今天" in text or "今日" in text:
        return anchor
    if "明天" in text or "明日" in text:
        return anchor + timedelta(days=1)

    return None


_LAST_YEAR = re.compile(r"\blast\s+year\b|去年", re.IGNORECASE)


def derive_time_facts(
    happened_on: date | None,
    text: str = "",
    *,
    today: date | None = None,
) -> list[str]:
    """产出可追加到 detail 的时间派生表述（短句列表）。"""
    facts: list[str] = []
    if happened_on is not None:
        facts.append(weekday_name(happened_on, lang="zh"))
        facts.append(weekday_name(happened_on, lang="en"))
        facts.append(happened_on.isoformat())

    anchor = happened_on or today
    if anchor is not None and text:
        resolved = resolve_relative(text, anchor)
        if resolved is not None and resolved != happened_on:
            facts.append(
                f"{resolved.isoformat()}（{weekday_name(resolved, lang='zh')}）"
            )
        # 「去年 / last year」→ 锚定年的上一年（LoCoMo 时序金标常为年份）
        if _LAST_YEAR.search(text):
            facts.append(str(anchor.year - 1))
            facts.append(f"year={anchor.year - 1} (last year)")

    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for f in facts:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def append_time_facts_to_detail(
    detail: str,
    happened_on: date | None,
    *,
    text: str = "",
    today: date | None = None,
    max_chars: int = 300,
) -> str:
    """把时间派生表述追加到 detail 尾部（不超 max_chars）。"""
    facts = derive_time_facts(happened_on, text, today=today)
    if not facts:
        return (detail or "")[:max_chars]
    marker = "time_facts="
    base = detail or ""
    # 避免重复追加
    if marker in base:
        return base[:max_chars]
    suffix = f" | {marker}{', '.join(facts)}"
    room = max_chars - len(suffix)
    if room <= 0:
        return suffix[-max_chars:]
    return (base[:room].rstrip() + suffix)[:max_chars]
