"""进程内聊天 demo 会话（清上下文只清短期历史，不动 wiki space）。

聊天记录落盘：logs/chat/{session_id}.jsonl（相对进程 cwd）。
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 相对启动目录；已在仓库 .gitignore 的 logs/
_CHAT_LOG_DIR = Path("logs") / "chat"
_FILE_LOCK = threading.Lock()


@dataclass
class ChatSession:
    id: str
    space_uid: str
    recall_method: str = "bm25"
    allow_memory_write: bool = True
    web_search: bool = False
    history: list[dict[str, str]] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)


_LOCK = threading.Lock()
_SESSIONS: dict[str, ChatSession] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_path(session_id: str) -> Path:
    return _CHAT_LOG_DIR / f"{session_id}.jsonl"


def _write_disk(session_id: str, entry: dict[str, Any]) -> None:
    """追加一行 JSONL；失败静默（不影响聊天）。"""
    try:
        path = _log_path(session_id)
        with _FILE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def append_log(sess: ChatSession, entry: dict[str, Any]) -> dict[str, Any]:
    """内存 + 磁盘各写一份。大字段 messages_to_llm 不落盘。"""
    row = dict(entry)
    row.setdefault("at", _now())
    row.setdefault("session_id", sess.id)
    row.setdefault("space_uid", sess.space_uid)
    sess.logs.append(row)
    disk = {k: v for k, v in row.items() if k != "messages_to_llm"}
    _write_disk(sess.id, disk)
    return row


def append_messages(
    sess: ChatSession, *, user: str, assistant: str
) -> None:
    """记录一回合纯对话文本（便于事后翻聊天记录）。"""
    append_log(
        sess,
        {"event": "message", "role": "user", "content": user},
    )
    append_log(
        sess,
        {"event": "message", "role": "assistant", "content": assistant},
    )


def create(
    *,
    space_uid: str | None = None,
    recall_method: str = "bm25",
    allow_memory_write: bool = True,
    web_search: bool = False,
) -> ChatSession:
    sid = uuid.uuid4().hex[:12]
    uid = space_uid or f"demo-{sid}"
    sess = ChatSession(
        id=sid,
        space_uid=uid,
        recall_method=recall_method,
        allow_memory_write=allow_memory_write,
        web_search=web_search,
    )
    with _LOCK:
        _SESSIONS[sid] = sess
    append_log(
        sess,
        {
            "event": "session_start",
            "recall_method": recall_method,
            "allow_memory_write": allow_memory_write,
            "web_search": web_search,
            "log_file": str(_log_path(sid).as_posix()),
        },
    )
    return sess


def get(session_id: str) -> ChatSession | None:
    with _LOCK:
        return _SESSIONS.get(session_id)


def clear_context(session_id: str) -> ChatSession | None:
    with _LOCK:
        sess = _SESSIONS.get(session_id)
        if sess is None:
            return None
        sess.history.clear()
        append_log(
            sess,
            {
                "event": "clear_context",
                "note": "短期对话历史已清空；wiki space 记忆保留",
            },
        )
        return sess
