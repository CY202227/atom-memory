"""memory_judge：除纯问候外每轮都调用 LLM。"""

from datetime import date

from atom_memory.demo.memory_judge import (
    judge_memory_save,
    sanitize_judge_note,
    should_run_judge,
    should_skip_judge,
)
from atom_memory.llm.base import ChatResult


class _CountingLLM:
    def __init__(self, text='{"save": true, "note": "应记", "reason": "test"}'):
        self.n = 0
        self.text = text
        self.last_user = ""

    def complete(self, system: str, user: str, response_format=None) -> ChatResult:
        del system, response_format
        self.n += 1
        self.last_user = user
        return ChatResult(
            text=self.text,
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_greeting_skips():
    assert should_skip_judge("你好")
    assert not should_run_judge("hello")


def test_news_question_still_runs_judge():
    q = "最近国际新闻有什么大事件"
    assert should_run_judge(q)


def test_correction_runs_judge():
    assert should_run_judge("鸣潮 不是新游戏")
    assert should_run_judge("为什么之前你不去搜索呢")


def test_health_disclosure_runs_judge():
    q = "我现在补的牙很多，不知道什么时候就得根管了"
    assert should_run_judge(q)


def test_judge_calls_llm_for_ordinary_turn():
    llm = _CountingLLM()
    d = judge_memory_save(
        llm,
        user_text="今天天气怎么样",
        assistant_text="挺好的",
    )
    assert not d.skipped
    assert d.save
    assert llm.n == 1
    assert "<current_time>" in llm.last_user


def test_judge_calls_llm_with_preference():
    llm = _CountingLLM()
    d = judge_memory_save(
        llm,
        user_text="叫我老张",
        assistant_text="好的老张",
    )
    assert not d.skipped
    assert d.save
    assert llm.n == 1


def test_greeting_still_skips_llm():
    llm = _CountingLLM()
    d = judge_memory_save(
        llm,
        user_text="你好",
        assistant_text="你好呀",
    )
    assert d.skipped
    assert d.reason == "skip:greeting_or_too_short"
    assert llm.n == 0


def test_sanitize_strips_clock_date_not_in_user():
    today = date(2026, 7, 27)
    note = "TA 于7月27日拔除智齿"
    out = sanitize_judge_note(note, "拔牙很疼", today=today)
    assert "7月27日" not in out
    assert "拔除" in out or "智齿" in out


def test_sanitize_keeps_date_user_stated():
    today = date(2026, 7, 27)
    note = "手术日期是2026-07-22"
    out = sanitize_judge_note(
        note, "纠正一下，手术是2026-07-22", today=today
    )
    assert "2026-07-22" in out


def test_judge_sanitizes_note_from_llm():
    today = date(2026, 7, 27)
    llm = _CountingLLM(
        text='{"save": true, "note": "TA 于7月27日拔除", "reason": "health"}'
    )
    d = judge_memory_save(
        llm,
        user_text="我拔牙了还疼",
        assistant_text="多休息",
        today=today,
    )
    assert d.save
    assert "7月27日" not in d.note
