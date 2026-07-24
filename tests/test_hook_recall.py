"""OpenAICompatLLM response_format 透传与降级。"""

import httpx
import pytest

from atom_memory.llm.base import LLMError
from atom_memory.llm.openai_compat import OpenAICompatLLM


def _ok_response(url):
    body = '{"operations": []}'
    return httpx.Response(
        200,
        request=httpx.Request("POST", url),
        json={
            "choices": [{"message": {"content": body}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        },
    )


def test_response_format_passthrough_and_fallback(monkeypatch):
    calls = []

    def post(url, **kw):
        payload = kw.get("json") or {}
        calls.append(payload)
        if "response_format" in payload:
            raise LLMError("no schema")
        return _ok_response(url)

    monkeypatch.setattr("atom_memory.llm.openai_compat.httpx.post", post)
    llm = OpenAICompatLLM("http://x/v1", "k", "m")
    res = llm.complete(
        "sys",
        "user",
        response_format={"type": "json_schema", "json_schema": {"name": "w", "schema": {}}},
    )
    assert res.text
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_response_format_supported_single_request(monkeypatch):
    calls = []

    def post(url, **kw):
        calls.append(kw.get("json") or {})
        return _ok_response(url)

    monkeypatch.setattr("atom_memory.llm.openai_compat.httpx.post", post)
    llm = OpenAICompatLLM("http://x/v1", "k", "m")
    llm.complete("s", "u", response_format={"type": "json_object"})
    assert len(calls) == 1


def test_response_format_both_attempts_fail(monkeypatch):
    def post(url, **kw):
        raise LLMError("down")

    monkeypatch.setattr("atom_memory.llm.openai_compat.httpx.post", post)
    llm = OpenAICompatLLM("http://x/v1", "k", "m")
    with pytest.raises(LLMError):
        llm.complete("s", "u", response_format={"type": "json_object"})
