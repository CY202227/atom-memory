"""服务配置。全部可由环境变量 / .env 覆盖。"""

from __future__ import annotations

import json
import os
from typing import Any

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


class Settings(BaseSettings):
    """服务配置。全部可由环境变量 / .env 覆盖。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ATOMMEM_",
        extra="ignore",
    )

    # sqlite 单文件起步；换 Postgres 只需改 URL，如
    # postgresql+psycopg://user:pass@127.0.0.1:5432/atom_memory
    database_url: str = "sqlite:///./atom_memory.db"

    # 固化/召回所用 LLM（OpenAI 兼容端点，vLLM 等均可）
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    # 经代理 + 本地大模型时，单次请求常要几分钟；评测建议 600+
    llm_timeout_seconds: float = 600.0
    # True=走系统 HTTP(S)_PROXY（经代理访问内网模型时保持 True）
    llm_trust_env: bool = True
    # false=在 extra_body 里关 Qwen3 思考（默认关，加速）
    llm_enable_thinking: bool = False
    # 可选 JSON，并入 /chat/completions（会与关思考字段合并）
    llm_extra_body_json: str = ""
    llm_extra_body: dict[str, Any] = Field(default_factory=dict)

    # 设置后所有请求须带 X-API-Key 头；留空则不鉴权（内网/本机模式）
    api_key: str = ""

    # 单次固化最多消费的 pending source 数
    consolidate_max_sources: int = 20

    # /chat demo：喂给主模型的短期 history 条数上限（user+assistant 合计）；
    # 0 表示不截断。长上下文变笨/费钱时可调低。
    chat_history_max_messages: int = 40

    # /chat demo：等后台固化回显的最长秒数
    chat_consolidate_wait_seconds: float = 8.0

    # /chat demo 联网搜索（DashScope WebSearch MCP SSE）
    dashscope_api_key: str = ""
    web_search_mcp_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/sse"
    )

    # 向量召回（空 api_base = 关闭；method=embedding/hybrid 时需配置）
    embedder_api_base: str = ""
    embedder_api_key: str = "EMPTY"
    embedder_model: str = ""
    embedder_timeout_seconds: float = 60.0
    embedder_trust_env: bool = True
    # 拼进向量的 detail 上限（statement 全量 + detail 截断）
    embed_detail_chars: int = 200

    @model_validator(mode="after")
    def _build_llm_extra_body(self) -> Settings:
        body: dict[str, Any] = {}
        raw = (self.llm_extra_body_json or "").strip()
        if raw:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("ATOMMEM_LLM_EXTRA_BODY_JSON must be a JSON object")
            body = dict(parsed)
        if not self.llm_enable_thinking:
            kwargs = dict(body.get("chat_template_kwargs") or {})
            kwargs["enable_thinking"] = False
            body["chat_template_kwargs"] = kwargs
            body.setdefault("enable_thinking", False)
        self.llm_extra_body = body
        return self


settings = Settings()
