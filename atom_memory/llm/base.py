"""LLM 是可替换轴：引擎只依赖此契约，不 import 任何 provider SDK。"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, code: str = "request_failed"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class ChatLLM(Protocol):
    def complete(
        self, system: str, user: str, response_format: dict | None = None
    ) -> ChatResult: ...
