import json
from typing import Any

import httpx

from .base import ChatResult, LLMError


class OpenAICompatLLM:
    """OpenAI 兼容 /chat/completions 适配器（vLLM、各中转均可）。

    ``trust_env=True``（默认）时会走系统 HTTP(S)_PROXY / ALL_PROXY，
    适合「必须经代理才能访问内网模型」的环境。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 300.0,
        *,
        trust_env: bool = True,
        extra_body: dict[str, Any] | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = float(timeout)
        self._extra_body = dict(extra_body or {})
        self.diagnostics: list[dict[str, Any]] = []
        # 评测会连打很多请求；复用连接。trust_env 控制是否读系统代理。
        self._client = httpx.Client(timeout=self._timeout, trust_env=trust_env)

    def _with_extra(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._extra_body:
            return payload
        # Request invariants and explicit retry budgets take precedence over extra_body.
        return {**self._extra_body, **payload}

    def complete(
        self, system: str, user: str, response_format: dict | None = None
    ) -> ChatResult:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        }
        if response_format is None:
            return self._request(payload)
        formatted = {**payload, "response_format": response_format}
        for attempt in range(2):
            try:
                result = self._request(formatted)
            except LLMError as exc:
                if attempt == 0 and (exc.code in ("timeout", "transport") or exc.status_code in (429, 502, 503, 504)):
                    continue
                if exc.code == "output_truncated" and attempt == 0:
                    field = "max_completion_tokens" if "max_completion_tokens" in self._extra_body else "max_tokens"
                    limit = formatted.get(field, self._extra_body.get(field))
                    if isinstance(limit, int) and 0 < limit < 16384:
                        formatted[field] = min(limit * 2, 16384)
                        continue
                if exc.code != "unsupported_response_format":
                    raise
                if formatted["response_format"].get("type") != "json_schema":
                    raise
                # An explicit capability rejection permits JSON mode, never unconstrained output.
                formatted["response_format"] = {"type": "json_object"}
                result = self._request(formatted)
            try:
                value = json.loads(result.text)
                if not isinstance(value, dict):
                    raise ValueError("expected JSON object")
                return result
            except (ValueError, TypeError) as exc:
                if attempt:
                    raise LLMError("Invalid structured JSON after 2 attempts", code="invalid_json") from exc
                formatted["messages"] = payload["messages"] + [{"role": "user", "content": "The previous output was invalid JSON. Return exactly one complete JSON object matching the requested schema. No prose or Markdown."}]
        raise AssertionError("unreachable")

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
    ) -> dict[str, Any]:
        """多轮对话；返回 assistant message 字典（可含 tool_calls）。"""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        data = self._request_raw(payload)
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected LLM response shape: {data}") from e

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
    ):
        """流式多轮对话：yield 文本增量（str）。"""
        for ev in self.chat_stream_events(messages, temperature=temperature):
            if ev["type"] == "token":
                yield ev["text"]

    def chat_stream_events(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
    ):
        """流式一轮：yield token；结束时 yield message（可含 tool_calls）。

        事件：
        - {"type": "token", "text": "..."}
        - {"type": "message", "message": {...}}
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        content_parts: list[str] = []
        tool_acc: dict[int, dict[str, Any]] = {}

        try:
            with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=self._with_extra(payload),
            ) as resp:
                if resp.status_code >= 400:
                    body = resp.read().decode("utf-8", errors="replace")[:400]
                    raise LLMError(
                        f"LLM stream failed: {resp.status_code} {resp.reason_phrase}"
                        f" body={body}"
                    )
                for line in resp.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data = line[5:].strip()
                    else:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    try:
                        choice = chunk["choices"][0]
                        delta = choice.get("delta") or {}
                    except (KeyError, IndexError, TypeError):
                        continue

                    piece = delta.get("content") or ""
                    if piece:
                        content_parts.append(piece)
                        # 一旦出现 tool_calls，正文不再当作最终回答流出
                        if not tool_acc:
                            yield {"type": "token", "text": piece}

                    for tc in delta.get("tool_calls") or []:
                        if not tool_acc:
                            yield {"type": "tools"}
                        idx = int(tc.get("index") or 0)
                        slot = tool_acc.setdefault(
                            idx,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        if tc.get("type"):
                            slot["type"] = tc["type"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] = (
                                slot["function"]["name"] + fn["name"]
                            )
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]

        except LLMError:
            raise
        except httpx.HTTPError as e:
            detail = ""
            resp = getattr(e, "response", None)
            if resp is not None:
                try:
                    detail = f" body={resp.text[:400]}"
                except Exception:  # noqa: BLE001
                    pass
            raise LLMError(f"LLM stream failed: {e}{detail}") from e

        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts) if content_parts else None,
        }
        if tool_acc:
            message["tool_calls"] = [tool_acc[i] for i in sorted(tool_acc)]
            for tc in message["tool_calls"]:
                if not tc.get("id"):
                    tc["id"] = f"call_{tc['function']['name'] or 'tool'}"
        yield {"type": "message", "message": message}

    def _request(self, payload: dict) -> ChatResult:
        data = self._request_raw(payload)
        try:
            choice = data["choices"][0]
            message = choice["message"]
            self.diagnostics.append({"format": payload.get("response_format", {}).get("type", "text"),
                "finish_reason": choice.get("finish_reason"),
                "content_chars": len(message.get("content") or "") if isinstance(message.get("content"), str) else 0,
                "completion_tokens": (data.get("usage") or {}).get("completion_tokens", 0)})
            self.diagnostics[:] = self.diagnostics[-256:]
            if choice.get("finish_reason") == "length":
                raise LLMError("Structured output truncated at token limit", code="output_truncated")
            if message.get("refusal") or choice.get("finish_reason") == "content_filter":
                raise LLMError("Model refused output", code="refusal")
            text = message["content"] or ""
            if not isinstance(text, str):
                raise TypeError("content must be text")
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("Unexpected LLM response shape", code="response_shape") from e
        usage = data.get("usage") or {}
        return ChatResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def _request_raw(self, payload: dict) -> dict[str, Any]:
        try:
            resp = self._client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=self._with_extra(payload),
            )
            if resp.status_code >= 400:
                body = resp.text.lower()
                unsupported = (resp.status_code in (400, 422) and any(token in body for token in ("response_format", "json_schema")) and any(token in body for token in ("unsupported", "not supported", "does not support")))
                raise LLMError(
                    f"LLM request failed: {resp.status_code}", status_code=resp.status_code,
                    code="unsupported_response_format" if unsupported else "request_failed",
                )
            try:
                data = resp.json()
            except ValueError as exc:
                raise LLMError("Provider returned non-JSON HTTP body", code="invalid_response_json") from exc
            if not isinstance(data, dict):
                raise LLMError("Provider returned non-object HTTP body", code="response_shape")
            return data
        except LLMError:
            raise
        except httpx.TimeoutException as e:
            raise LLMError(
                f"LLM request timed out after {self._timeout}s", code="timeout"
            ) from e
        except httpx.HTTPError as e:
            raise LLMError("LLM transport request failed", code="transport") from e

    def close(self) -> None:
        self._client.close()
