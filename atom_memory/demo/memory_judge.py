"""chat demo：回合后记忆裁判（独立短调用，不进主对话 history）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from ..consolidation.engine import parse_json_object
from ..llm.base import ChatLLM, LLMError

_JUDGE_SYSTEM = """\
你是数字人的记忆写入裁判（独立子代理）。主对话模型可能漏调 memory_save。
根据本回合用户话与助手回复，判断是否有值得长期写入 wiki 的稳定信息。

应保存（save=true）——即便只出现一次，只要可复用：
- 人设/口癖/角色；称呼偏好
- 用户偏好、习惯、价值观倾向（含支出态度，如「每月花约 1/4 收入健身并纠结是否浪费」）
- 兴趣爱好、喜欢/讨厌、长期偏好
- 健康/身体状况、医疗自我披露与诊断结论、就医计划
- 重要关系、职业/项目、稳定身份事实
- 用户明确要求记住的内容
- 用户纠正助手的错误事实（应记成教训，避免再犯）
- 用户对助手行为的约束/期望（如「不确定先联网搜」「别瞎编发售日」）
- 一次性对话里出现的可复用用户侧事实或偏好（不要因「只聊一次」就 save=false）

不应保存（save=false）——仅这些收窄：
- 对已披露/已入库事实的纯机制或科普追问，且本轮用户无新事实、无新偏好、无新计划
  （如已知牙本质敏感后只问「为何含着疼」「为何喝冰水不疼」）
- 纯问候、纯外部新闻/百科问答且无用户侧信息、无纠正与行为约束
- 助手单方面病理猜测或建议，用户未确认的臆测

## 日期硬规则
- 用户消息里会给出「当前时间」。禁止把「今天/刚才」写成具体公历日期，除非用户原文已给出该日期。
- 用户纠正了日期时，note 必须保留用户给出的日期原文，禁止改成对话当天。
- 若没有明确可记的日期，note 不要写「于X月X日…」。

只输出一个 JSON 对象：
{"save": true|false, "note": "若 save 则为简洁第三人称认识陈述（可空）", "reason": "一句话理由"}
note 要用认识断言口吻（如「老张补牙较多，担心很快需要根管」；
「时效事实须先联网搜索再答，勿凭印象把老游当新发售」），不要写「用户说了…」。
"""

_SKIP_RE = re.compile(
    r"^(你好|您好|在吗|嗨|hi|hello|hey|早上好|晚安|谢谢|感谢)[\s!！.。?？]*$",
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


def should_run_judge(user_text: str) -> bool:
    """是否值得花一次 LLM 裁判（除纯问候外每轮都跑）。"""
    return not should_skip_judge(user_text)


def sanitize_judge_note(
    note: str,
    user_text: str,
    today: date | None = None,
) -> str:
    """去掉 note 里由「对话时钟」臆造、但用户原文未提及的今日日期。"""
    if not note:
        return note
    today = today or date.today()
    user = user_text or ""
    out = note

    patterns: list[re.Pattern[str]] = [
        re.compile(rf"{today.year}年\s*{today.month}月\s*{today.day}日"),
        re.compile(rf"{today.month}月\s*{today.day}日"),
        re.compile(rf"{today.month}月\s*{today.day:02d}日"),
        re.compile(re.escape(today.isoformat())),
        re.compile(rf"(?<!\d){today.month:02d}-{today.day:02d}(?!\d)"),
    ]
    for pat in patterns:
        if pat.search(out) and not pat.search(user):
            out = pat.sub("", out)

    # 「于…拔除」类残留介词
    out = re.sub(r"于\s*(?=拔|做|去|到|在)", "", out)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip(" ，,。；;")


def judge_memory_save(
    llm: ChatLLM,
    *,
    user_text: str,
    assistant_text: str,
    today: date | None = None,
) -> JudgeDecision:
    """同步调用：是否把本回合线索写入长期记忆。"""
    if should_skip_judge(user_text):
        return JudgeDecision(
            save=False,
            note="",
            reason="skip:greeting_or_too_short",
            skipped=True,
        )

    today = today or date.today()
    user_payload = (
        f"<current_time>{today.isoformat()}</current_time>\n"
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
    if save and note:
        note = sanitize_judge_note(note, user_text, today=today)
    if not save:
        note = ""
    return JudgeDecision(
        save=save,
        note=note,
        reason=reason or ("worth_saving" if save else "not_worth_saving"),
        raw=res.text[:1000],
    )
