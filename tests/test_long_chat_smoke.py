"""连续长聊烟雾：夹杂噪声后关键事实仍在库且可召回（默认不清上下文语义）。"""

from __future__ import annotations

import json

from atom_memory.consolidation.canonical import (
    CANONICAL_PERSONA,
    CANONICAL_PREFERRED_NAME,
)


def _uid(client) -> str:
    return client.post(
        "/spaces", json={"owner_id": "long", "subject_id": "chat"}
    ).json()["uid"]


def _ingest(client, uid: str, content: str, *, salience: float = 0.8) -> int:
    return client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": content, "salience": salience},
    ).json()["id"]


def test_long_chat_facts_survive_noise_rounds(client, fake_llm):
    """设人设/称呼/健康 → 多轮空固化噪声 → 召回仍命中关键 atom。"""
    uid = _uid(client)

    # --- 人设 ---
    s1 = _ingest(client, uid, "用户：你是口癖喵的猫娘。AI：好的喵。")
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "self",
                        "key": "persona-catgirl",
                        "statement": "我是口癖带喵的猫娘",
                        "detail": "",
                        "change_reason": "人设",
                        "source_ids": [s1],
                    }
                ]
            },
            ensure_ascii=False,
        )
    ]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == (
        "succeeded"
    )

    # --- 称呼 ---
    s2 = _ingest(client, uid, "用户：叫我老张。AI：好的老张喵。")
    fake_llm.responses = [
        json.dumps({"read": [CANONICAL_PERSONA]}),
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": "user-preference-name",
                        "statement": "用户希望被称呼为老张",
                        "detail": "",
                        "change_reason": "称呼",
                        "source_ids": [s2],
                    }
                ]
            },
            ensure_ascii=False,
        ),
    ]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == (
        "succeeded"
    )

    # --- 健康 ---
    s3 = _ingest(
        client,
        uid,
        "用户：我补牙很多，怕很快根管。AI：记下了喵。",
        salience=0.85,
    )
    fake_llm.responses = [
        json.dumps({"read": [CANONICAL_PREFERRED_NAME]}),
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": "dental-health",
                        "statement": "老张补牙较多，担心很快需要根管",
                        "detail": "关注再生牙进展",
                        "change_reason": "健康披露",
                        "source_ids": [s3],
                    }
                ]
            },
            ensure_ascii=False,
        ),
    ]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == (
        "succeeded"
    )

    # --- 噪声轮：新闻闲聊固化为空操作（模拟长聊夹杂）---
    for i in range(8):
        sid = _ingest(
            client,
            uid,
            f"用户：今天国际新闻第{i}条怎样？AI：一般般喵。",
            salience=0.1,
        )
        fake_llm.responses = [
            json.dumps({"read": []}),
            json.dumps({"operations": []}),
        ]
        run = client.post(f"/spaces/{uid}/consolidate", json={}).json()
        assert run["status"] == "succeeded"
        # 噪声源应被跳过或消费但不动关键 atoms
        _ = sid

    index = {e["key"]: e for e in client.get(f"/spaces/{uid}/atoms").json()["results"]}
    assert CANONICAL_PERSONA in index or "persona-catgirl" in index
    # persona-catgirl 应被规范到 persona（首轮无已有时）
    assert CANONICAL_PERSONA in index
    assert CANONICAL_PREFERRED_NAME in index
    assert "dental-health" in index
    # 不应因噪声暴涨
    assert len(index) <= 5

    # 近重复再写一次健康 → 应并入 dental-health
    s4 = _ingest(
        client,
        uid,
        "用户：我补了好多牙真怕根管。AI：记得喵。",
        salience=0.85,
    )
    fake_llm.responses = [
        json.dumps({"read": ["dental-health"]}),
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": "teeth-worry-again",
                        "statement": "老张补牙很多，担心很快就要根管",
                        "detail": "仍关注牙齿",
                        "change_reason": "再次提及",
                        "source_ids": [s4],
                    }
                ]
            },
            ensure_ascii=False,
        ),
    ]
    run = client.post(f"/spaces/{uid}/consolidate", json={}).json()
    assert run["status"] == "succeeded"
    assert "dental-health" in run["atoms_touched"]
    assert "teeth-worry-again" not in run["atoms_touched"]
    index2 = {e["key"] for e in client.get(f"/spaces/{uid}/atoms").json()["results"]}
    assert "teeth-worry-again" not in index2
    assert "dental-health" in index2

    # BM25 召回身体相关
    r = client.post(
        f"/spaces/{uid}/recall",
        json={"query": "我的身体状况和牙齿", "method": "bm25", "max_atoms": 5},
    ).json()
    keys = {h["key"] for h in r["hits"]}
    assert "dental-health" in keys

    # 身份相关：至少称呼或人设在库（自动召回侧 sticky 在 chat_demo；此处测 API）
    r2 = client.post(
        f"/spaces/{uid}/recall",
        json={"query": "老张 称呼", "method": "bm25", "max_atoms": 5},
    ).json()
    keys2 = {h["key"] for h in r2["hits"]}
    assert CANONICAL_PREFERRED_NAME in keys2 or "dental-health" in keys2


def test_history_window_helper():
    from atom_memory.api.routes import chat_demo
    from atom_memory.config import settings

    hist = [{"role": "user", "content": str(i)} for i in range(50)]
    old = settings.chat_history_max_messages
    try:
        settings.chat_history_max_messages = 10
        fed, truncated = chat_demo._history_for_llm(hist)
        assert truncated
        assert len(fed) == 10
        assert fed[0]["content"] == "40"
        settings.chat_history_max_messages = 0
        fed2, truncated2 = chat_demo._history_for_llm(hist)
        assert not truncated2
        assert len(fed2) == 50
    finally:
        settings.chat_history_max_messages = old
