"""聊天失忆调试页：短期上下文 vs 长期 wiki 记忆（流式 + 工具化记忆）。

GET /chat — 前端
POST /chat/api/sessions — 开局
POST /chat/api/sessions/{id}/message/stream — SSE：工具循环→流式生成
POST /chat/api/sessions/{id}/clear-context — 只清短期历史
POST /chat/api/sessions/{id}/consolidate — 手动固化
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import Session

from ... import demo
from ...config import settings
from ...consolidation.engine import ConsolidationEngine
from ...db import engine as db_engine
from ...db import get_session
from ...llm import ChatLLM
from ...llm.base import LLMError
from ...llm.openai_compat import OpenAICompatLLM
from ...locks import space_write_lock
from ...models import AtomKind, AtomStatus, SourceKind, SourceStatus
from ...recall import clip_by_budget, fallback_hits, is_inventory_query, render_statement_block
from ...recall.base import RecallHit
from ...recall.llm import RecallError
from ...repositories import atom_repo, source_repo, space_repo
from ..deps import (
    build_recall_strategy,
    get_engine,
    get_llm,
    require_api_key,
)

router = APIRouter(tags=["chat-demo"])

_CHAT_HTML = Path(__file__).resolve().parent.parent.parent / "web" / "chat.html"

_CORE_KINDS = frozenset({AtomKind.self})
_CORE_MAX = 2
# 称呼偏好：BM25 对「我是谁」几乎必空，必须常驻
_STICKY_KEYS = frozenset({"user-preferred-name"})
# 仅身份问句（清单式问句走 is_inventory_query，避免与 fallback 正则双份）
_IDENTITY_QUERY_RE = re.compile(
    r"(我是谁|我叫什么|叫我什么|我的名字|你还记得我|记得我吗|"
    r"who am i|what('s| is) my name|do you remember me)",
    re.IGNORECASE,
)
# 与 API 默认对齐，不另起一套
_RECALL_BUDGET = 400
_RECALL_MAX = 5
_EXPAND_BUDGET = 1200
_EXPAND_MAX_KEYS = 10

_SYSTEM = (
    "你是一个有长期记忆的助手（本页为调试 demo，非正式 SDK）。"
    "当前回合 user 带 <context>：current_time 与自动召回的短 statement；"
    "回答「现在/今天」以 current_time 为准。"
    "读：优先用 recalled_memory；不够再 memory_search / memory_expand。"
    "写：遇到可复用的用户事实或偏好可调 memory_save（入库后后台固化）；"
    "漏存时由后台裁判补判。不要假装记得库里没有的内容。"
    "若有 web_search：仅实时/外部事实且知识不足时再调。"
)

_MEMORY_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": (
            "换关键词再搜长期记忆（statement）。"
            "仅当自动召回不够时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词或短问句",
                }
            },
            "required": ["query"],
        },
    },
}

_MEMORY_EXPAND_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_expand",
        "description": "按 key 展开 detail；statement 不够时再调。",
        "parameters": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要展开的 atom key 列表",
                }
            },
            "required": ["keys"],
        },
    },
}

_MEMORY_SAVE_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_save",
        "description": (
            "写入值得长期保留的用户事实或偏好（先入库，后台固化）。"
            "问候、无新信息的闲聊不要调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "要记住的内容（简洁陈述）",
                },
                "reason": {
                    "type": "string",
                    "description": "为何值得保留（可选）",
                },
            },
            "required": ["note"],
        },
    },
}

_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "联网检索；仅实时/外部事实且知识不足时调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词或问句",
                }
            },
            "required": ["query"],
        },
    },
}


def _now_local() -> str:
    try:
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("Asia/Shanghai"))
    except Exception:  # noqa: BLE001
        now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %A %H:%M")


def _history_for_llm(
    history: list[dict[str, str]],
) -> tuple[list[dict[str, str]], bool]:
    """按配置截断短期 history；返回 (切片, 是否发生截断)。"""
    limit = int(settings.chat_history_max_messages or 0)
    if limit <= 0 or len(history) <= limit:
        return list(history), False
    return list(history[-limit:]), True


def _render_user_turn(
    user_text: str,
    *,
    memory_block: str = "",
    current_time: str | None = None,
) -> str:
    lines = [
        "<context>",
        f"  <current_time>{escape(current_time or _now_local())}</current_time>",
    ]
    if memory_block.strip():
        lines.append(memory_block.strip())
    lines.append("</context>")
    return (
        "\n".join(lines)
        + f"\n<user_message>\n{escape(user_text)}\n</user_message>"
    )


class StartRequest(BaseModel):
    space_uid: str | None = None
    recall_method: Literal["bm25", "fuzzy", "llm"] = "bm25"
    allow_memory_write: bool = True
    web_search: bool = False


class StartResponse(BaseModel):
    session_id: str
    space_uid: str
    recall_method: str
    allow_memory_write: bool
    web_search: bool
    web_search_configured: bool
    llm_model: str
    llm_base_url: str


class MessageRequest(BaseModel):
    text: str = Field(min_length=1)
    allow_memory_write: bool | None = None
    web_search: bool | None = None


class SessionState(BaseModel):
    session_id: str
    space_uid: str
    recall_method: str
    allow_memory_write: bool
    web_search: bool
    history: list[dict[str, str]]
    logs: list[dict[str, Any]]
    atoms_count: int
    pending_sources: int


@router.get("/chat", response_class=HTMLResponse, include_in_schema=False)
def chat_page() -> str:
    return _CHAT_HTML.read_text(encoding="utf-8")


@router.post(
    "/chat/api/sessions",
    response_model=StartResponse,
    dependencies=[Depends(require_api_key)],
    include_in_schema=False,
)
def start_session(
    payload: StartRequest,
    session: Session = Depends(get_session),
):
    uid = payload.space_uid or f"demo-{uuid.uuid4().hex[:10]}"
    space = space_repo.get_or_create(session, uid=uid)

    chat = demo.sessions.create(
        space_uid=space.uid,
        recall_method=payload.recall_method,
        allow_memory_write=payload.allow_memory_write,
        web_search=payload.web_search,
    )
    return StartResponse(
        session_id=chat.id,
        space_uid=chat.space_uid,
        recall_method=chat.recall_method,
        allow_memory_write=chat.allow_memory_write,
        web_search=chat.web_search,
        web_search_configured=bool(settings.dashscope_api_key),
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
    )


@router.get(
    "/chat/api/sessions/{session_id}",
    response_model=SessionState,
    dependencies=[Depends(require_api_key)],
    include_in_schema=False,
)
def get_session_state(
    session_id: str,
    session: Session = Depends(get_session),
):
    chat = demo.sessions.get(session_id)
    if chat is None:
        raise HTTPException(404, "session not found")
    space = space_repo.get_by_uid(session, chat.space_uid)
    atoms_n = len(atom_repo.list_active(session, space.id)) if space else 0
    pending_n = 0
    if space:
        pending_n = len(
            source_repo.list_by_space(session, space.id, status=SourceStatus.pending)
        )
    return SessionState(
        session_id=chat.id,
        space_uid=chat.space_uid,
        recall_method=chat.recall_method,
        allow_memory_write=chat.allow_memory_write,
        web_search=chat.web_search,
        history=list(chat.history),
        logs=list(chat.logs),
        atoms_count=atoms_n,
        pending_sources=pending_n,
    )


@router.post(
    "/chat/api/sessions/{session_id}/clear-context",
    response_model=SessionState,
    dependencies=[Depends(require_api_key)],
    include_in_schema=False,
)
def clear_context(
    session_id: str,
    session: Session = Depends(get_session),
):
    chat = demo.sessions.clear_context(session_id)
    if chat is None:
        raise HTTPException(404, "session not found")
    return get_session_state(session_id, session)


@router.post(
    "/chat/api/sessions/{session_id}/consolidate",
    dependencies=[Depends(require_api_key)],
    include_in_schema=False,
)
def consolidate_now(
    session_id: str,
    session: Session = Depends(get_session),
    engine: ConsolidationEngine = Depends(get_engine),
):
    chat = demo.sessions.get(session_id)
    if chat is None:
        raise HTTPException(404, "session not found")
    space = space_repo.get_by_uid(session, chat.space_uid)
    if space is None:
        raise HTTPException(404, "space not found")
    with space_write_lock(space.id):
        run = engine.run(
            session,
            space,
            trigger="chat-demo",
            max_sources=settings.consolidate_max_sources,
        )
    entry = demo.sessions.append_log(
        chat,
        {
            "event": "consolidate",
            "status": (
                run.status.value if hasattr(run.status, "value") else run.status
            ),
            "atoms_touched": list(run.atoms_touched or []),
            "error": run.error,
        },
    )
    return entry


@router.post(
    "/chat/api/sessions/{session_id}/message/stream",
    dependencies=[Depends(require_api_key)],
    include_in_schema=False,
)
async def send_message_stream(
    session_id: str,
    payload: MessageRequest,
    llm: ChatLLM = Depends(get_llm),
    engine: ConsolidationEngine = Depends(get_engine),
):
    """SSE 流：phase / memory / search / token / done / error。"""
    chat = demo.sessions.get(session_id)
    if chat is None:
        raise HTTPException(404, "session not found")
    if not isinstance(llm, OpenAICompatLLM):
        raise HTTPException(500, "chat demo requires OpenAICompatLLM")

    text = payload.text.strip()
    allow_write = (
        chat.allow_memory_write
        if payload.allow_memory_write is None
        else payload.allow_memory_write
    )
    do_search = chat.web_search if payload.web_search is None else payload.web_search
    chat.allow_memory_write = allow_write
    chat.web_search = do_search

    async def event_gen():
        try:
            yield _sse("phase", {"phase": "recall"})
            auto = _light_auto_recall(
                chat.space_uid, chat.recall_method, text, llm
            )
            yield _sse(
                "recall",
                {
                    "auto_hits": auto["auto_hits"],
                    "core_keys": auto["core_keys"],
                    "active_atoms_n": auto["active_atoms_n"],
                    "semantic_fallback": auto["semantic_fallback"],
                    "context_block": auto["memory_block"],
                    "error": auto.get("error"),
                },
            )

            memory_trace: list[dict[str, Any]] = []
            search_trace: list[dict[str, Any]] = []
            save_waiters: list[dict[str, Any]] = []

            user_payload = _render_user_turn(
                text,
                memory_block=auto["memory_block"] or "",
            )
            history_fed, history_truncated = _history_for_llm(chat.history)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": _SYSTEM},
                *history_fed,
                {"role": "user", "content": user_payload},
            ]

            tools = [_MEMORY_SEARCH_TOOL, _MEMORY_EXPAND_TOOL]
            tool_names = ["memory_search", "memory_expand"]
            if allow_write:
                tools.append(_MEMORY_SAVE_TOOL)
                tool_names.append("memory_save")
            if do_search:
                tools.append(_WEB_SEARCH_TOOL)
                tool_names.append("web_search")

            pieces: list[str] = []
            try:
                async for ev in _generate_with_tools(
                    llm,
                    messages,
                    space_uid=chat.space_uid,
                    session_id=chat.id,
                    chat=chat,
                    recall_method=chat.recall_method,
                    tools=tools,
                    engine=engine,
                    memory_trace=memory_trace,
                    search_trace=search_trace,
                    save_waiters=save_waiters,
                ):
                    if ev["type"] == "phase":
                        yield _sse("phase", {"phase": ev["phase"]})
                    elif ev["type"] == "memory":
                        yield _sse("memory", ev["data"])
                    elif ev["type"] == "search":
                        yield _sse("search", ev["data"])
                    elif ev["type"] == "reset_tokens":
                        pieces.clear()
                        yield _sse("reset", {})
                    elif ev["type"] == "token":
                        pieces.append(ev["text"])
                        yield _sse("token", {"text": ev["text"]})
            except LLMError as e:
                yield _sse("error", {"message": f"LLM failed: {e}"})
                return

            reply = "".join(pieces).strip()
            chat.history.append({"role": "user", "content": text})
            chat.history.append({"role": "assistant", "content": reply})
            demo.sessions.append_messages(chat, user=text, assistant=reply)

            # 主模型漏 save 时：独立裁判子调用决定是否入库
            had_save = any(
                c.get("tool") == "memory_save" and c.get("ok")
                for c in memory_trace
            )
            judge_trace: dict[str, Any] | None = None
            if allow_write and not had_save:
                yield _sse("phase", {"phase": "judge"})
                decision = await asyncio.to_thread(
                    demo.memory_judge.judge_memory_save,
                    llm,
                    user_text=text,
                    assistant_text=reply,
                )
                judge_trace = {
                    "ok": decision.error is None,
                    "tool": "memory_judge",
                    "save": decision.save,
                    "note": decision.note or None,
                    "reason": decision.reason,
                    "skipped": decision.skipped,
                    "error": decision.error,
                }
                memory_trace.append(judge_trace)
                yield _sse("memory", judge_trace)

                if decision.save and decision.note:
                    try:
                        yield _sse("phase", {"phase": "persist"})
                        queued = _memory_save_ingest(
                            space_uid=chat.space_uid,
                            session_id=chat.id,
                            note=decision.note,
                            reason=decision.reason or "memory_judge",
                            via="memory_judge",
                            salience=0.8,
                        )
                        waiter: dict[str, Any] = {
                            "event": threading.Event(),
                            "done": None,
                        }
                        save_waiters.append(waiter)
                        _schedule_consolidate(
                            space_uid=chat.space_uid,
                            space_id=queued["space_id"],
                            session_id=chat.id,
                            source_id=queued["source_id"],
                            note=queued["note"],
                            engine=engine,
                            chat=chat,
                            waiter=waiter,
                        )
                        entry = {
                            "ok": True,
                            "tool": "memory_save",
                            "via": "memory_judge",
                            "source_id": queued["source_id"],
                            "note": queued["note"],
                            "reason": queued["reason"],
                            "status": "queued",
                        }
                        memory_trace.append(entry)
                        yield _sse("memory", entry)
                    except Exception as e:  # noqa: BLE001
                        entry = {
                            "ok": False,
                            "tool": "memory_save",
                            "via": "memory_judge",
                            "error": str(e),
                        }
                        memory_trace.append(entry)
                        yield _sse("memory", entry)

            # 快写慢固：流式结束后短暂等待已排队的固化，便于本轮 debug
            for waiter in save_waiters:
                done_ev: threading.Event = waiter["event"]
                finished = await asyncio.to_thread(
                    done_ev.wait, settings.chat_consolidate_wait_seconds
                )
                if finished:
                    done_entry = waiter.get("done")
                    if done_entry:
                        memory_trace.append(done_entry)
                        yield _sse("memory", done_entry)

            saves = [
                c
                for c in memory_trace
                if c.get("tool") in ("memory_save", "memory_save_done")
            ]
            done_saves = [c for c in memory_trace if c.get("tool") == "memory_save_done"]
            judge_saves = [c for c in saves if c.get("via") == "memory_judge"]
            debug = {
                "at": _now(),
                "event": "turn",
                "space_uid": chat.space_uid,
                "recall_method": chat.recall_method,
                "sticky_keys": auto["core_keys"],
                "core_keys": auto["core_keys"],
                "auto_hits": auto["auto_hits"],
                "hits": auto["auto_hits"],
                "semantic_fallback": auto["semantic_fallback"],
                "memory_calls": memory_trace,
                "user_turn_payload": user_payload,
                "injection": "auto_light_recall",
                "active_atoms_n": auto["active_atoms_n"],
                "allow_memory_write": allow_write,
                "web_search_enabled": do_search,
                "web_search_calls": search_trace,
                "short_term_history_len": len(chat.history),
                "history_fed_len": len(history_fed),
                "history_truncated": history_truncated,
                "messages_to_llm": messages,
                "llm": {
                    "model": settings.llm_model,
                    "base_url": settings.llm_base_url,
                    "stream": True,
                    "tools": tool_names,
                },
                "memory_saves": saves,
                "memory_judge": judge_trace,
                "judge_saves": judge_saves,
                "consolidate": (
                    done_saves[-1].get("consolidate") if done_saves else None
                ),
                "note_clear_context": (
                    "点「清除上下文」只丢短期 history；"
                    "长期记忆靠自动召回 + memory_save / memory_judge 写入"
                ),
            }
            demo.sessions.append_log(chat, debug)
            yield _sse(
                "done",
                {
                    "reply": reply,
                    "debug": debug,
                    "history_len": len(chat.history),
                    "logs": list(chat.logs),
                },
            )
        except Exception as e:  # noqa: BLE001
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _norm_statement(text: str) -> str:
    return "".join((text or "").split()).lower()


def _dedupe_hits(hits: list[RecallHit]) -> list[RecallHit]:
    """归一化后相同或互相包含的 statement 去重，保留先出现的。"""
    out: list[RecallHit] = []
    norms: list[str] = []
    for h in hits:
        n = _norm_statement(h.atom.statement)
        if not n:
            continue
        if any(n == x or n in x or x in n for x in norms):
            continue
        norms.append(n)
        out.append(h)
    return out


def _retrieve_hits(
    atoms: list,
    method: str,
    query: str,
    llm: ChatLLM,
    *,
    force_semantic: bool = False,
) -> tuple[list[RecallHit], bool, str | None]:
    strategy = build_recall_strategy(method, llm)
    recall_error = None
    hits_raw: list[RecallHit] = []
    semantic_fallback = False
    try:
        outcome = strategy.retrieve(atoms, query, max_atoms=_RECALL_MAX)
        hits_raw = list(outcome.hits)
    except RecallError as e:
        recall_error = str(e)

    need_semantic = (not hits_raw or force_semantic) and atoms and method != "llm"
    if need_semantic:
        try:
            from ...recall.llm import LlmRecall

            secondary = LlmRecall(llm).retrieve(atoms, query, _RECALL_MAX)
            if secondary.hits:
                # force 时合并，避免丢掉已有 BM25 命中
                if hits_raw and force_semantic:
                    seen = {h.atom.key for h in hits_raw}
                    for h in secondary.hits:
                        if h.atom.key not in seen:
                            hits_raw.append(h)
                            seen.add(h.atom.key)
                else:
                    hits_raw = list(secondary.hits)
                semantic_fallback = True
        except RecallError as e:
            recall_error = (
                f"{recall_error}; semantic: {e}" if recall_error else str(e)
            )
    return hits_raw, semantic_fallback, recall_error


def _light_auto_recall(
    space_uid: str, method: str, query: str, llm: ChatLLM
) -> dict[str, Any]:
    """self + 称呼 sticky；BM25/语义检索补其余；身份问句强制语义。"""
    with Session(db_engine) as session:
        space = space_repo.get_by_uid(session, space_uid)
        if space is None:
            raise HTTPException(404, "space not found")
        atoms = atom_repo.list_active(session, space.id)
        by_key = {a.key: a for a in atoms}

        core: list[RecallHit] = []
        for key in _STICKY_KEYS:
            atom = by_key.get(key)
            if atom is not None:
                core.append(RecallHit(atom=atom, score=None))
        self_core = [
            RecallHit(atom=a, score=None)
            for a in atoms
            if a.kind in _CORE_KINDS and a.key not in _STICKY_KEYS
        ][:_CORE_MAX]
        core.extend(self_core)
        core_keys = [h.atom.key for h in core]
        have = set(core_keys)

        force_semantic = bool(_IDENTITY_QUERY_RE.search(query or ""))
        inventory = is_inventory_query(query) or force_semantic
        # 身份问句且称呼未在 sticky（尚未固化）时也走语义
        hits_raw, semantic_fallback, recall_error = _retrieve_hits(
            atoms,
            method,
            query,
            llm,
            force_semantic=force_semantic,
        )
        if (inventory or not hits_raw) and atoms:
            exclude = have | {h.atom.key for h in hits_raw}
            fill_n = max(0, _RECALL_MAX - len(have) - len(hits_raw))
            if inventory and not hits_raw:
                fill_n = max(fill_n, _RECALL_MAX - len(have))
            hits_raw = list(hits_raw) + fallback_hits(
                atoms, exclude_keys=exclude, limit=fill_n
            )
        extra = [h for h in hits_raw if h.atom.key not in have]
        merged = _dedupe_hits(core + extra)
        merged = clip_by_budget(merged, budget_chars=_RECALL_BUDGET)[:_RECALL_MAX]
        auto_hits = [
            {
                "key": h.atom.key,
                "kind": h.atom.kind.value,
                "statement": h.atom.statement,
                "score": h.score,
                "core": h.atom.key in have,
            }
            for h in merged
        ]
        return {
            "memory_block": render_statement_block(merged),
            "auto_hits": auto_hits,
            "core_keys": core_keys,
            "active_atoms_n": len(atoms),
            "semantic_fallback": semantic_fallback,
            "error": recall_error,
        }


def _memory_search(
    space_uid: str, method: str, query: str, llm: ChatLLM
) -> dict[str, Any]:
    with Session(db_engine) as session:
        space = space_repo.get_by_uid(session, space_uid)
        if space is None:
            raise HTTPException(404, "space not found")
        atoms = atom_repo.list_active(session, space.id)
        hits_raw, semantic_fallback, recall_error = _retrieve_hits(
            atoms, method, query, llm
        )
        if (is_inventory_query(query) or not hits_raw) and atoms:
            exclude = {h.atom.key for h in hits_raw}
            fill_n = max(0, _RECALL_MAX - len(hits_raw))
            hits_raw = list(hits_raw) + fallback_hits(
                atoms, exclude_keys=exclude, limit=fill_n
            )
        hits_raw = _dedupe_hits(hits_raw)
        hits_raw = clip_by_budget(hits_raw, budget_chars=_RECALL_BUDGET)[
            :_RECALL_MAX
        ]
        hits = [
            {
                "key": h.atom.key,
                "kind": h.atom.kind.value,
                "statement": h.atom.statement,
                "score": h.score,
            }
            for h in hits_raw
        ]
        return {
            "query": query,
            "method": method,
            "hits": hits,
            "semantic_fallback": semantic_fallback,
            "error": recall_error,
            "active_atoms_n": len(atoms),
        }


def _memory_expand(space_uid: str, keys: list[str]) -> dict[str, Any]:
    with Session(db_engine) as session:
        space = space_repo.get_by_uid(session, space_uid)
        if space is None:
            raise HTTPException(404, "space not found")
        hits: list[dict[str, Any]] = []
        missing: list[str] = []
        used = 0
        for key in keys[:_EXPAND_MAX_KEYS]:
            atom = atom_repo.get_by_key(session, space.id, key)
            if atom is None or atom.status != AtomStatus.active:
                missing.append(key)
                continue
            detail = atom.detail or atom.statement
            cost = len(atom.statement) + len(detail)
            if hits and used + cost > _EXPAND_BUDGET:
                missing.append(key)
                continue
            hits.append(
                {
                    "key": atom.key,
                    "kind": atom.kind.value,
                    "statement": atom.statement,
                    "detail": detail,
                }
            )
            used += cost
        return {"keys": keys[:_EXPAND_MAX_KEYS], "hits": hits, "missing": missing}


def _memory_save_ingest(
    *,
    space_uid: str,
    session_id: str,
    note: str,
    reason: str,
    via: str = "memory_save",
    salience: float = 0.9,
) -> dict[str, Any]:
    """同步落 Source，立刻返回 queued。"""
    content = note.strip()
    if reason.strip():
        content = f"{content}\n（原因：{reason.strip()}）"
    kind = (
        SourceKind.turn
        if via in ("memory_judge", "auto_interest")
        else SourceKind.correction
    )
    with Session(db_engine) as session:
        space = space_repo.get_by_uid(session, space_uid)
        if space is None:
            raise HTTPException(404, "space not found")
        source = source_repo.add(
            session,
            space_id=space.id,
            kind=kind,
            content=content,
            salience=salience,
            external_ref={
                "system": "chat-demo",
                "session_id": session_id,
                "via": via,
            },
        )
        return {
            "source_id": source.id,
            "space_id": space.id,
            "note": note.strip(),
            "reason": reason.strip() or None,
            "status": "queued",
            "via": via,
        }


def _schedule_consolidate(
    *,
    space_uid: str,
    space_id: int,
    session_id: str,
    source_id: int,
    note: str,
    engine: ConsolidationEngine,
    chat: demo.sessions.ChatSession,
    waiter: dict[str, Any],
) -> None:
    """后台固化；结果写入 waiter['done'] 与 session logs。"""

    def work() -> None:
        try:
            with Session(db_engine) as session:
                space = space_repo.get_by_uid(session, space_uid)
                if space is None:
                    raise RuntimeError(f"space not found: {space_uid}")
                with space_write_lock(space_id):
                    run = engine.run(
                        session,
                        space,
                        trigger="memory_save",
                        max_sources=settings.consolidate_max_sources,
                    )
                consolidate = {
                    "status": (
                        run.status.value
                        if hasattr(run.status, "value")
                        else run.status
                    ),
                    "atoms_touched": list(run.atoms_touched or []),
                    "error": run.error,
                    "run_id": run.id,
                }
                entry = {
                    "ok": True,
                    "tool": "memory_save_done",
                    "source_id": source_id,
                    "note": note,
                    "consolidate": consolidate,
                    "at": _now(),
                }
        except Exception as e:  # noqa: BLE001
            entry = {
                "ok": False,
                "tool": "memory_save_done",
                "source_id": source_id,
                "note": note,
                "error": str(e),
                "at": _now(),
            }
        waiter["done"] = entry
        demo.sessions.append_log(
            chat,
            {
                "event": "memory_save_done",
                **entry,
            },
        )
        waiter["event"].set()

    threading.Thread(target=work, daemon=True, name="memory-save-consolidate").start()


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _parse_tool_args(raw_args: Any) -> dict[str, Any]:
    try:
        if isinstance(raw_args, str):
            return json.loads(raw_args) if raw_args.strip() else {}
        return raw_args or {}
    except ValueError:
        return {}


async def _generate_with_tools(
    llm: OpenAICompatLLM,
    messages: list[dict[str, Any]],
    *,
    space_uid: str,
    session_id: str,
    chat: demo.sessions.ChatSession,
    recall_method: str,
    tools: list[dict[str, Any]],
    engine: ConsolidationEngine,
    memory_trace: list[dict[str, Any]],
    search_trace: list[dict[str, Any]],
    save_waiters: list[dict[str, Any]],
    max_tool_rounds: int = 4,
):
    """流式工具循环：边生成边出 token；有 tool_calls 则执行后继续。"""
    for _round_i in range(max_tool_rounds):
        yield {"type": "phase", "phase": "generate"}
        msg: dict[str, Any] | None = None
        for ev in llm.chat_stream_events(
            messages,
            temperature=0.7,
            tools=tools,
            tool_choice="auto",
        ):
            if ev["type"] == "tools":
                # 本轮改为工具调用：清掉可能已流出的正文
                yield {"type": "reset_tokens"}
            elif ev["type"] == "token":
                yield {"type": "token", "text": ev["text"]}
            elif ev["type"] == "message":
                msg = ev["message"]

        if msg is None:
            return

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return

        messages.append(
            {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            }
        )

        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            call_id = tc.get("id") or "call"
            args = _parse_tool_args(fn.get("arguments") or "{}")

            if name == "memory_search":
                query = str(args.get("query") or "").strip()
                yield {"type": "phase", "phase": "memory"}
                if not query:
                    entry = {
                        "ok": False,
                        "tool": "memory_search",
                        "error": "empty query",
                    }
                    memory_trace.append(entry)
                    yield {"type": "memory", "data": entry}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": "memory_search failed: empty query",
                        }
                    )
                    continue
                result = _memory_search(space_uid, recall_method, query, llm)
                entry = {"ok": True, "tool": "memory_search", **result}
                memory_trace.append(entry)
                yield {"type": "memory", "data": entry}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(
                            {
                                "hits": result["hits"],
                                "semantic_fallback": result["semantic_fallback"],
                                "error": result["error"],
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
                continue

            if name == "memory_expand":
                raw_keys = args.get("keys") or []
                if isinstance(raw_keys, str):
                    keys = [raw_keys.strip()] if raw_keys.strip() else []
                else:
                    keys = [str(k).strip() for k in raw_keys if str(k).strip()]
                yield {"type": "phase", "phase": "memory"}
                if not keys:
                    entry = {
                        "ok": False,
                        "tool": "memory_expand",
                        "error": "empty keys",
                    }
                    memory_trace.append(entry)
                    yield {"type": "memory", "data": entry}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": "memory_expand failed: empty keys",
                        }
                    )
                    continue
                result = _memory_expand(space_uid, keys)
                entry = {"ok": True, "tool": "memory_expand", **result}
                memory_trace.append(entry)
                yield {"type": "memory", "data": entry}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(
                            {
                                "hits": result["hits"],
                                "missing": result["missing"],
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
                continue

            if name == "memory_save":
                note = str(args.get("note") or "").strip()
                reason = str(args.get("reason") or "").strip()
                yield {"type": "phase", "phase": "persist"}
                if not note:
                    entry = {
                        "ok": False,
                        "tool": "memory_save",
                        "error": "empty note",
                    }
                    memory_trace.append(entry)
                    yield {"type": "memory", "data": entry}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": "memory_save failed: empty note",
                        }
                    )
                    continue
                try:
                    queued = _memory_save_ingest(
                        space_uid=space_uid,
                        session_id=session_id,
                        note=note,
                        reason=reason,
                    )
                    waiter: dict[str, Any] = {
                        "event": threading.Event(),
                        "done": None,
                    }
                    save_waiters.append(waiter)
                    _schedule_consolidate(
                        space_uid=space_uid,
                        space_id=queued["space_id"],
                        session_id=session_id,
                        source_id=queued["source_id"],
                        note=note,
                        engine=engine,
                        chat=chat,
                        waiter=waiter,
                    )
                    entry = {
                        "ok": True,
                        "tool": "memory_save",
                        "source_id": queued["source_id"],
                        "note": queued["note"],
                        "reason": queued["reason"],
                        "status": "queued",
                    }
                    memory_trace.append(entry)
                    yield {"type": "memory", "data": entry}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(
                                {
                                    "saved": True,
                                    "source_id": queued["source_id"],
                                    "status": "queued",
                                    "note": "已入库，固化在后台进行；可继续回答用户",
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    entry = {
                        "ok": False,
                        "tool": "memory_save",
                        "error": str(e),
                        "note": note,
                    }
                    memory_trace.append(entry)
                    yield {"type": "memory", "data": entry}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": f"memory_save failed: {e}",
                        }
                    )
                continue

            if name == "web_search":
                query = str(args.get("query") or "").strip()
                if not query:
                    err = f"unsupported or empty tool call: {name!r}"
                    search_trace.append({"ok": False, "error": err, "name": name})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": err,
                        }
                    )
                    continue
                yield {"type": "phase", "phase": "search"}
                try:
                    result = await demo.web_search.search_async(query)
                    text = (result.get("text") or "").strip()
                    entry = {
                        "ok": True,
                        "tool": result.get("tool"),
                        "query": query,
                        "arguments": result.get("arguments"),
                        "text": text[:4000],
                    }
                    search_trace.append(entry)
                    yield {"type": "search", "data": entry}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": text[:8000] or "(empty search result)",
                        }
                    )
                except demo.web_search.WebSearchError as e:
                    entry = {"ok": False, "error": str(e), "query": query}
                    search_trace.append(entry)
                    yield {"type": "search", "data": entry}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": f"search failed: {e}",
                        }
                    )
                continue

            err = f"unsupported tool: {name!r}"
            memory_trace.append({"ok": False, "tool": name, "error": err})
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": err}
            )

    # 工具轮次耗尽：无工具流式终答
    yield {"type": "phase", "phase": "generate"}
    for piece in llm.chat_stream(messages, temperature=0.7):
        yield {"type": "token", "text": piece}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
