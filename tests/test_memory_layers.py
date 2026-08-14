"""L0–L3 分层：写路径标注、读时推断、layered vs flat 召回。"""

from __future__ import annotations

import json

from atom_memory.memory_layers import (
    LAYER_L1,
    LAYER_L2,
    LAYER_L3,
    effective_layer,
    infer_layer,
)
from atom_memory.models import Atom, AtomKind, AtomStatus
from datetime import datetime, timezone


def _uid(client) -> str:
    return client.post(
        "/spaces",
        json={"owner_id": "layers", "subject_id": "bench"},
    ).json()["uid"]


def test_infer_layer_rules():
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    a1 = Atom(
        id=1,
        space_id=1,
        kind=AtomKind.event,
        key="evt-1",
        statement="去了公园",
        status=AtomStatus.active,
        memory_layer=1,
        created_at=now,
        updated_at=now,
    )
    assert infer_layer(a1) == LAYER_L1
    assert effective_layer(a1, derived_ids={1}) == LAYER_L2

    a3 = Atom(
        id=2,
        space_id=1,
        kind=AtomKind.person,
        key="user-preferred-name",
        statement="叫我老张",
        status=AtomStatus.active,
        memory_layer=1,
        created_at=now,
        updated_at=now,
    )
    assert infer_layer(a3) == LAYER_L3
    assert effective_layer(a3) == LAYER_L3


def test_consolidate_marks_l1_and_persona_l3(client, fake_llm):
    uid = _uid(client)
    r = client.post(
        f"/spaces/{uid}/sources",
        json={
            "kind": "turn",
            "content": "用户：请叫我老张。\nAI：好的。",
            "salience": 0.9,
        },
    )
    sid = r.json()["id"]
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": "user-preferred-name",
                        "statement": "称呼偏好：老张",
                        "detail": "",
                        "change_reason": "称呼",
                        "source_ids": [sid],
                    },
                    {
                        "op": "upsert",
                        "kind": "event",
                        "key": "met-today",
                        "statement": "今天见了面",
                        "detail": "",
                        "change_reason": "事件",
                        "source_ids": [sid],
                    },
                ]
            },
            ensure_ascii=False,
        )
    ]
    run = client.post(f"/spaces/{uid}/consolidate", json={}).json()
    assert run["status"] == "succeeded"

    atoms = client.get(
        f"/spaces/{uid}/atoms", params={"page_size": 50}
    ).json()["results"]
    by_key = {a["key"]: a for a in atoms}
    assert by_key["user-preferred-name"]["memory_layer"] == LAYER_L3
    assert by_key["met-today"]["memory_layer"] == LAYER_L1


def test_layered_recall_prioritizes_l3(client, fake_llm):
    uid = _uid(client)
    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "seed", "salience": 0.9},
    )
    sid = r.json()["id"]
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": "user-preferred-name",
                        "statement": "称呼偏好：阿花",
                        "detail": "请叫我阿花",
                        "change_reason": "称呼",
                        "source_ids": [sid],
                    },
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "likes-tea",
                        "statement": "喜欢绿茶",
                        "detail": "",
                        "change_reason": "偏好",
                        "source_ids": [sid],
                    },
                    {
                        "op": "upsert",
                        "kind": "event",
                        "key": "noise-event",
                        "statement": "随便聊聊天气",
                        "detail": "",
                        "change_reason": "闲聊",
                        "source_ids": [sid],
                    },
                ]
            },
            ensure_ascii=False,
        )
    ]
    client.post(f"/spaces/{uid}/consolidate", json={})

    layered = client.post(
        f"/spaces/{uid}/recall",
        json={
            "query": "天气怎么样",
            "method": "bm25",
            "max_atoms": 2,
            "budget_chars": 400,
            "policy": "layered",
            "include_recent_sources": False,
        },
    ).json()
    keys = [h["key"] for h in layered["hits"]]
    assert "user-preferred-name" in keys
    assert keys[0] == "user-preferred-name"

    flat = client.post(
        f"/spaces/{uid}/recall",
        json={
            "query": "天气怎么样",
            "method": "bm25",
            "max_atoms": 2,
            "budget_chars": 400,
            "policy": "flat",
            "include_recent_sources": False,
        },
    ).json()
    assert flat["hits"]  # flat 仍可返回


def test_synthesize_writes_l2(client, fake_llm):
    uid = _uid(client)
    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "hike and camp", "salience": 0.8},
    )
    sid = r.json()["id"]
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "hobby-hike",
                        "statement": "喜欢徒步",
                        "detail": "",
                        "change_reason": "a",
                        "source_ids": [sid],
                    },
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "hobby-camp",
                        "statement": "喜欢露营",
                        "detail": "",
                        "change_reason": "b",
                        "source_ids": [sid],
                    },
                ]
            }
        )
    ]
    client.post(f"/spaces/{uid}/consolidate", json={})

    fake_llm.responses = [
        json.dumps({"read": ["hobby-hike", "hobby-camp"]}),
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "outdoor-lifestyle",
                        "statement": "偏爱户外生活方式",
                        "detail": "",
                        "confidence": 0.7,
                        "change_reason": "综合",
                        "derived_from": ["hobby-hike", "hobby-camp"],
                    }
                ]
            }
        ),
    ]
    run = client.post(f"/spaces/{uid}/synthesize", json={}).json()
    assert run["status"] == "succeeded"
    atom = client.get(
        f"/spaces/{uid}/atoms/outdoor-lifestyle"
    ).json()
    assert atom["memory_layer"] == LAYER_L2


def test_persona_endpoint_marks_l3(client, fake_llm):
    """无有效 LLM 产出时，persona 端点仍把 canonical/self 标成 L3。"""
    uid = _uid(client)
    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "persona", "salience": 0.9},
    )
    sid = r.json()["id"]
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "self",
                        "key": "persona",
                        "statement": "我是体贴的助手",
                        "detail": "",
                        "change_reason": "人设",
                        "source_ids": [sid],
                    }
                ]
            }
        )
    ]
    client.post(f"/spaces/{uid}/consolidate", json={})

    # consolidate 已把 persona 标为 L3；端点再跑一遍应 succeeded
    fake_llm.responses = [json.dumps({"operations": []})]
    run = client.post(f"/spaces/{uid}/persona", json={}).json()
    assert run["status"] == "succeeded"
    atom = client.get(f"/spaces/{uid}/atoms/persona").json()
    assert atom["memory_layer"] == LAYER_L3
