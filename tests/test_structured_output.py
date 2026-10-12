import json

import httpx
import pytest

from atom_memory.consolidation.engine import WritePlan, _json_schema_format
from atom_memory.llm.base import LLMError
from atom_memory.llm.openai_compat import OpenAICompatLLM


def response(content, reason="stop", refusal=None):
    return httpx.Response(200, json={"choices": [{"finish_reason": reason,
        "message": {"content": content, "refusal": refusal}}]})


@pytest.mark.parametrize("status", [401, 500])
def test_transport_error_does_not_disable_schema(monkeypatch, status):
    calls = []
    def post(self, url, **kw):
        calls.append(kw["json"])
        return httpx.Response(status, json={"error": "request failed"})
    monkeypatch.setattr(httpx.Client, "post", post)
    with pytest.raises(LLMError) as caught:
        OpenAICompatLLM("http://test/v1", "key", "model").complete("s", "u", _json_schema_format("write", WritePlan))
    assert caught.value.status_code == status
    assert len(calls) == 1
    assert calls[0]["response_format"]["json_schema"]["strict"]


def test_transient_error_retries_with_same_schema(monkeypatch):
    calls = []
    def post(self, url, **kw):
        calls.append(kw["json"])
        return httpx.Response(503, json={"error": "unavailable"}) if len(calls) == 1 else response('{"operations": []}')
    monkeypatch.setattr(httpx.Client, "post", post)
    OpenAICompatLLM("http://test/v1", "key", "model").complete("s", "u", _json_schema_format("write", WritePlan))
    assert len(calls) == 2
    assert calls[0]["response_format"] == calls[1]["response_format"]


def test_invalid_json_retries_without_dropping_format(monkeypatch):
    calls = []
    def post(self, url, **kw):
        calls.append(kw["json"])
        return response('{"operations":' if len(calls) == 1 else '{"operations": []}')
    monkeypatch.setattr(httpx.Client, "post", post)
    result = OpenAICompatLLM("http://test/v1", "key", "model").complete("s", "u", {"type": "json_object"})
    assert json.loads(result.text) == {"operations": []}
    assert len(calls) == 2
    assert all(call["response_format"]["type"] == "json_object" for call in calls)


@pytest.mark.parametrize("output,code", [
    (response('{"operations":', "length"), "output_truncated"),
    (response(None, refusal="no"), "refusal"),
    (httpx.Response(200, text="<html>gateway</html>"), "invalid_response_json"),
])
def test_failure_reason_is_explicit(monkeypatch, output, code):
    monkeypatch.setattr(httpx.Client, "post", lambda *a, **kw: output)
    with pytest.raises(LLMError) as caught:
        OpenAICompatLLM("http://test/v1", "key", "model").complete("s", "u", {"type": "json_object"})
    assert caught.value.code == code


def test_strict_schema_recurses_into_defs():
    fmt = _json_schema_format("write", WritePlan)
    assert fmt["json_schema"]["strict"] is True
    def check(node):
        if isinstance(node, dict):
            assert "default" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)
    check(fmt["json_schema"]["schema"])


def test_truncation_retry_increases_budget_and_keeps_schema(monkeypatch):
    calls = []
    def post(self, url, **kw):
        calls.append(kw["json"])
        return response('{"operations":', "length") if len(calls) == 1 else response('{"operations": []}')
    monkeypatch.setattr(httpx.Client, "post", post)
    client = OpenAICompatLLM("http://test/v1", "key", "model", extra_body={"max_tokens": 1024, "response_format": {"type": "text"}})
    client.complete("s", "u", _json_schema_format("write", WritePlan))
    assert [call["max_tokens"] for call in calls] == [1024, 2048]
    assert all(call["response_format"]["json_schema"]["strict"] for call in calls)


def test_missing_operations_cannot_silently_skip_sources(client, fake_llm):
    uid = client.post("/spaces", json={"owner_id": "test", "subject_id": "test"}).json()["uid"]
    client.post(f"/spaces/{uid}/sources", json={"kind": "turn", "content": "Alice moved to Paris in 2023.", "salience": 0.2})
    fake_llm.responses = ['{"wrong_field": []}', '{"wrong_field": []}']
    run = client.post(f"/spaces/{uid}/consolidate", json={}).json()
    assert run["status"] == "failed"
    assert client.get(f"/spaces/{uid}/sources").json()[0]["status"] == "pending"


def test_schema_retry_recovers_before_consuming_source(client, fake_llm):
    uid = client.post("/spaces", json={"owner_id": "retry", "subject_id": "retry"}).json()["uid"]
    source = client.post(f"/spaces/{uid}/sources", json={"kind": "turn", "content": "Alice lives in Paris."}).json()["id"]
    fake_llm.responses = ['{"other": []}', json.dumps({"operations": [{"op": "upsert", "kind": "person", "key": "alice-home", "statement": "Alice lives in Paris.", "source_ids": [source]}]})]
    run = client.post(f"/spaces/{uid}/consolidate", json={}).json()
    assert run["status"] == "succeeded"
    assert len(fake_llm.calls) == 2
    assert client.get(f"/spaces/{uid}/sources").json()[0]["status"] == "consolidated"
