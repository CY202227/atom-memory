"""chat demo：回合后记忆裁判（独立短调用，不进主对话 history）。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..consolidation.engine import parse_json_object
from ..llm.base import ChatLLM, LLMError

_JUDGE_SYSTEM = """\
你是数字人的记忆写入裁判（独立子代理）。主对话模型可能漏调 memory_save。
根据本回合用户话与助手回复，判断是否有值得长期写入 wiki 的稳定信息。

应保存（save=true）的例子：
- 人设/口癖/角色；称呼偏好
- 兴趣爱好、喜欢/讨厌、习惯
- 健康/身体状况、医疗相关自我披露（补牙、过敏、慢性病等）
- 重要关系、职业/项目、稳定身份事实
- 用户明确要求记住的内容

不应保存（save=false）：
- 纯问候、一次性闲聊、新闻问答、抽象观点辩论
- 已明显只在追问外部事实、无新的用户侧稳定线索
- 助手单方面猜测、用户未确认的臆测

只输出一个 JSON 对象：
{"save": true|false, "note": "若 save 则为简洁第三人称认识陈述（可空）", "reason": "一句话理由"}
note 要用认识断言口吻（如「老张补牙较多，担心很快需要根管」），不要写「用户说了…」。
"""

_SKIP_RE = re.compile(
    r"^(你好|您好|在吗|嗨|hi|hello|hey|早上好|晚安|谢谢|感谢)[\s!！.。?？]*$",
    re.IGNORECASE,
)

# 轻门闩：无稳定自我披露线索则不调用 LLM（降本、降噪）
_SIGNAL_RE = re.compile(
    r"(叫我|称呼|记住|别忘|喜欢|讨厌|爱好|兴趣|习惯|偏好|"
    r"口癖|人设|角色|你是|"
    r"过敏|补牙|根管|牙疼|身体|慢性|吃药|医院|病|"
    r"我的(猫|狗|孩子|儿子|女儿|老婆|老公|对象|公司|项目|工作)|"
    r"我是|我叫|我在做|工作是|职业是|"
    r"\bi like\b|\bi love\b|\bi hate\b|call me|remember (this|that|me))",
    re.IGNORECASE,
)


@dataclass
class JudgeDecision:
    save: bool
    note: str
    reason: str
    raw: str = ""
    error: str | None = None
    skipped: bool = False


def should_skip_judge(user_text: str) -> bool:
    """问候/过短：直接跳过。"""
    text = (user_text or "").strip()
    if len(text) < 2:
        return True
    return bool(_SKIP_RE.match(text))


def has_memory_signal(user_text: str) -> bool:
    """是否像含值得记的稳定线索（门闩，非最终裁判）。"""
    text = (user_text or "").strip()
    if not text:
        return False
    if _SIGNAL_RE.search(text):
        return True
    # 较长第一人称陈述也可能含事实
    if "我" in text and len(text) >= 16:
        return True
    return False


def should_run_judge(user_text: str) -> bool:
    """是否值得花一次 LLM 裁判。"""
    if should_skip_judge(user_text):
        return False
    return has_memory_signal(user_text)


def judge_memory_save(
    llm: ChatLLM,
    *,
    user_text: str,
    assistant_text: str,
) -> JudgeDecision:
    """同步调用：是否把本回合线索写入长期记忆。"""
    if should_skip_judge(user_text):
        return JudgeDecision(
            save=False,
            note="",
            reason="skip:greeting_or_too_short",
            skipped=True,
        )
    if not has_memory_signal(user_text):
        return JudgeDecision(
            save=False,
            note="",
            reason="skip:no_memory_signal",
            skipped=True,
        )

    user_payload = (
        f"<user_message>\n{user_text.strip()}\n</user_message>\n"
        f"<assistant_reply>\n{(assistant_text or '').strip()[:2000]}\n"
        f"</assistant_reply>"
    )
    try:
        res = llm.complete(
            _JUDGE_SYSTEM,
            user_payload,
            response_format={"type": "json_object"},
        )
        data = parse_json_object(res.text)
    except (LLMError, ValueError, TypeError) as e:
        return JudgeDecision(
            save=False,
            note="",
            reason="judge_failed",
            error=str(e),
        )

    save = bool(data.get("save"))
    note = str(data.get("note") or "").strip()
    reason = str(data.get("reason") or "").strip()
    if save and not note:
        note = user_text.strip()[:300]
    if not save:
        note = ""
    return JudgeDecision(
        save=save,
        note=note,
        reason=reason or ("worth_saving" if save else "not_worth_saving"),
        raw=res.text[:1000],
    )
