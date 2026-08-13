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
        # 评测会连打很多请求；复用连接。trust_env 控制是否读系统代理。
        self._client = httpx.Client(timeout=self._timeout, trust_env=trust_env)

    def _with_extra(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._extra_body:
            return payload
        return {**payload, **self._extra_body}

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
        if response_format is not None:
            try:
                return self._request({**payload, "response_format": response_format})
            except LLMError:
                pass
        return self._request(payload)

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
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected LLM response shape: {data}") from e
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
                raise LLMError(
                    f"LLM request failed: {resp.status_code} {resp.reason_phrase}"
                    f" body={resp.text[:400]}"
                )
            return resp.json()
        except LLMError:
            raise
        except httpx.TimeoutException as e:
            raise LLMError(
                f"LLM request timed out after {self._timeout}s "
                f"(url={self._base_url}; 经代理访问时请加大 "
                f"ATOMMEM_LLM_TIMEOUT_SECONDS，并确认代理与模型服务正常)"
            ) from e
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request failed: {e}") from e

    def close(self) -> None:
        self._client.close()
