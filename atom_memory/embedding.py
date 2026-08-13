"""embedder 适配：OpenAI 兼容 /embeddings（可替换轴）。

model 名即向量身份标（model_tag）：召回时与落库行不符则失效重算。
"""

from __future__ import annotations

from typing import Any

import httpx


class EmbedderError(Exception):
    """embedder 请求或响应异常。"""


class OpenAICompatEmbedder:
    """批量 embed；顺序与输入一致。"""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        *,
        trust_env: bool = True,
    ):
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self.model_tag = model
        self._timeout = float(timeout)
        self._client = httpx.Client(timeout=self._timeout, trust_env=trust_env)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = self._client.post(
                f"{self._api_base}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self.model_tag, "input": texts},
            )
            if resp.status_code >= 400:
                raise EmbedderError(
                    f"embedder request failed: {resp.status_code} "
                    f"{resp.reason_phrase} body={resp.text[:400]}"
                )
            data: dict[str, Any] = resp.json()
        except EmbedderError:
            raise
        except httpx.HTTPError as e:
            raise EmbedderError(f"embedder request failed: {e}") from e
        try:
            rows = sorted(data["data"], key=lambda d: d["index"])
            vectors = [row["embedding"] for row in rows]
        except (KeyError, TypeError) as e:
            raise EmbedderError(f"unexpected embedder response shape: {data}") from e
        if len(vectors) != len(texts):
            raise EmbedderError(
                f"embedder returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors

    def close(self) -> None:
        self._client.close()
