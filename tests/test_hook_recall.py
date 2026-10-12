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

    def post(self, url, **kw):
        payload = kw.get("json") or {}
        calls.append(payload)
        if payload.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, json={"error": "response_format json_schema is not supported"})
        return _ok_response(url)

    monkeypatch.setattr("atom_memory.llm.openai_compat.httpx.Client.post", post)
    llm = OpenAICompatLLM("http://x/v1", "k", "m")
    res = llm.complete(
        "sys",
        "user",
        response_format={"type": "json_schema", "json_schema": {"name": "w", "schema": {}}},
    )
    assert res.text
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert calls[1]["response_format"] == {"type": "json_object"}


def test_response_format_supported_single_request(monkeypatch):
    calls = []

    def post(self, url, **kw):
        calls.append(kw.get("json") or {})
        return _ok_response(url)

    monkeypatch.setattr("atom_memory.llm.openai_compat.httpx.Client.post", post)
    llm = OpenAICompatLLM("http://x/v1", "k", "m")
    llm.complete("s", "u", response_format={"type": "json_object"})
    assert len(calls) == 1


def test_response_format_both_attempts_fail(monkeypatch):
    def post(self, url, **kw):
        raise LLMError("down")

    monkeypatch.setattr("atom_memory.llm.openai_compat.httpx.Client.post", post)
    llm = OpenAICompatLLM("http://x/v1", "k", "m")
    with pytest.raises(LLMError):
        llm.complete("s", "u", response_format={"type": "json_object"})
