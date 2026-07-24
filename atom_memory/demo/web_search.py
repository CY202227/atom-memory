"""DashScope WebSearch MCP（SSE）客户端：供 /chat demo 联网检索。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from ..config import settings


class WebSearchError(Exception):
    pass


def _mcp_http_client(
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """直连 MCP；忽略 HTTP(S)_PROXY / 系统代理，避免代理 TLS 失败。"""
    kwargs: dict[str, Any] = {
        "follow_redirects": True,
        "trust_env": False,
    }
    if timeout is None:
        kwargs["timeout"] = httpx.Timeout(30.0, read=300.0)
    else:
        kwargs["timeout"] = timeout
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)


def search(query: str, *, timeout: float = 45.0) -> dict[str, Any]:
    """同步封装：在事件循环外调用时用 asyncio.run。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(search_async(query, timeout=timeout))
    # 已在 async 上下文：不允许 asyncio.run，调用方应直接 await
    raise WebSearchError("use search_async inside running event loop")


async def search_async(query: str, *, timeout: float = 45.0) -> dict[str, Any]:
    """经 MCP SSE 调联网搜索；返回 {tool, arguments, text, raw}。"""
    api_key = (settings.dashscope_api_key or "").strip()
    url = (settings.web_search_mcp_url or "").strip()
    if not api_key:
        raise WebSearchError("ATOMMEM_DASHSCOPE_API_KEY 未配置")
    if not url:
        raise WebSearchError("ATOMMEM_WEB_SEARCH_MCP_URL 未配置")

    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
    except ImportError as e:
        raise WebSearchError("缺少 mcp 包：pip install mcp") from e

    headers = {"Authorization": f"Bearer {api_key}"}
    query = query.strip()
    if not query:
        raise WebSearchError("empty query")

    async def _run() -> dict[str, Any]:
        async with sse_client(
            url,
            headers=headers,
            httpx_client_factory=_mcp_http_client,
        ) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                listed = await session.list_tools()
                tools = list(listed.tools or [])
                if not tools:
                    raise WebSearchError("MCP 未返回任何 tool")
                tool = _pick_search_tool(tools)
                args = _build_args(tool, query)
                result = await session.call_tool(tool.name, args)
                text = _result_text(result)
                return {
                    "tool": tool.name,
                    "arguments": args,
                    "text": text,
                    "tools_available": [t.name for t in tools],
                }

    try:
        return await asyncio.wait_for(_run(), timeout=timeout)
    except TimeoutError as e:
        raise WebSearchError(f"web search timeout (>{timeout}s)") from e
    except WebSearchError:
        raise
    except httpx.ConnectError as e:
        raise WebSearchError(
            "无法连接 DashScope WebSearch MCP（网络/防火墙/证书）。"
            "可取消勾选「允许联网」，或检查能否访问 dashscope.aliyuncs.com"
        ) from e
    except Exception as e:  # noqa: BLE001
        raise WebSearchError(f"web search failed: {e}") from e


def _pick_search_tool(tools: list[Any]) -> Any:
    preferred = ("web_search", "search", "bing_search", "google_search")
    by_name = {t.name: t for t in tools}
    for name in preferred:
        if name in by_name:
            return by_name[name]
    for t in tools:
        if "search" in (t.name or "").lower():
            return t
    return tools[0]


def _build_args(tool: Any, query: str) -> dict[str, Any]:
    """按 schema 猜查询字段名；常见为 query / q / search_term。"""
    schema = getattr(tool, "inputSchema", None) or {}
    props = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(props, dict) and props:
        for key in ("query", "q", "search_term", "keyword", "text"):
            if key in props:
                return {key: query}
        # 取第一个 string 字段
        for key, meta in props.items():
            if isinstance(meta, dict) and meta.get("type", "string") == "string":
                return {key: query}
        first = next(iter(props))
        return {first: query}
    return {"query": query}


def _result_text(result: Any) -> str:
    parts: list[str] = []
    content = getattr(result, "content", None) or []
    for block in content:
        t = getattr(block, "text", None)
        if t:
            parts.append(str(t))
        elif hasattr(block, "model_dump"):
            parts.append(json.dumps(block.model_dump(), ensure_ascii=False))
        else:
            parts.append(str(block))
    if parts:
        return "\n".join(parts)
    data = getattr(result, "data", None)
    if data is not None:
        return json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
    return str(result)
