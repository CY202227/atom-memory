"""memory_judge 门闩：无信号不调用 LLM。"""

from atom_memory.demo.memory_judge import (
    has_memory_signal,
    judge_memory_save,
    should_run_judge,
    should_skip_judge,
)
from atom_memory.llm.base import ChatResult


class _CountingLLM:
    def __init__(self):
        self.n = 0

    def complete(self, system: str, user: str, response_format=None) -> ChatResult:
        self.n += 1
        return ChatResult(
            text='{"save": true, "note": "应记", "reason": "test"}',
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_greeting_skips():
    assert should_skip_judge("你好")
    assert not should_run_judge("hello")


def test_news_question_no_signal():
    q = "最近国际新闻有什么大事件"
    assert not has_memory_signal(q)
    assert not should_run_judge(q)


def test_health_disclosure_has_signal():
    q = "我现在补的牙很多，不知道什么时候就得根管了"
    assert has_memory_signal(q)
    assert should_run_judge(q)


def test_judge_skips_llm_without_signal():
    llm = _CountingLLM()
    d = judge_memory_save(
        llm,
        user_text="今天天气怎么样",
        assistant_text="挺好的",
    )
    assert d.skipped
    assert d.reason == "skip:no_memory_signal"
    assert llm.n == 0


def test_judge_calls_llm_with_signal():
    llm = _CountingLLM()
    d = judge_memory_save(
        llm,
        user_text="叫我老张",
        assistant_text="好的老张",
    )
    assert not d.skipped
    assert d.save
    assert llm.n == 1
