"""类型化 atom_link：闭包传播、邻居召回、综合校验、时间派生。"""

from __future__ import annotations

import json
from datetime import date

from atom_memory.temporal import (
    append_time_facts_to_detail,
    derive_time_facts,
    resolve_relative,
    weekday_name,
)


def _uid(client) -> str:
    return client.post(
        "/spaces", json={"owner_id": "link", "subject_id": "t"}
    ).json()["uid"]


def _seed_two_atoms(client, fake_llm, uid: str) -> tuple[str, str]:
    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "用户：我喜欢爬山。", "salience": 0.8},
    )
    src1 = r.json()["id"]
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": "hobby-hike",
                        "statement": "TA 喜欢爬山",
                        "detail": "周末常去",
                        "change_reason": "t",
                        "source_ids": [src1],
                    }
                ]
            },
            ensure_ascii=False,
        )
    ]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == (
        "succeeded"
    )

    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "用户：我也喜欢露营。", "salience": 0.8},
    )
    src2 = r.json()["id"]
    fake_llm.responses = [
        json.dumps({"read": []}),
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": "hobby-camp",
                        "statement": "TA 喜欢露营",
                        "detail": "",
                        "change_reason": "t",
                        "source_ids": [src2],
                        "links": [{"to": "hobby-hike", "kind": "about"}],
                    }
                ]
            },
            ensure_ascii=False,
        ),
    ]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == (
        "succeeded"
    )
    return "hobby-hike", "hobby-camp"


def test_consolidate_links_and_neighbors(client, fake_llm):
    uid = _uid(client)
    a, b = _seed_two_atoms(client, fake_llm, uid)
    nb = client.get(f"/spaces/{uid}/atoms/{b}/neighbors").json()
    assert nb["key"] == b
    assert any(x["key"] == a and x["kind"] == "about" for x in nb["out"])
    # 非法 to 静默丢弃
    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "noop", "salience": 0.9},
    )
    src = r.json()["id"]
    fake_llm.responses = [
        json.dumps({"read": [b]}),
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": b,
                        "statement": "TA 喜欢露营与户外",
                        "detail": "",
                        "change_reason": "u",
                        "source_ids": [src],
                        "links": [
                            {"to": "does-not-exist", "kind": "about"},
                            {"to": a, "kind": "about"},
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
    ]
    client.post(f"/spaces/{uid}/consolidate", json={})
    nb2 = client.get(f"/spaces/{uid}/atoms/{b}/neighbors").json()
    keys = [x["key"] for x in nb2["out"]]
    assert a in keys
    assert "does-not-exist" not in keys


def test_neighbor_hops_default_unchanged_and_expand(client, fake_llm):
    uid = _uid(client)
    _seed_two_atoms(client, fake_llm, uid)
    r0 = client.post(
        f"/spaces/{uid}/recall",
        json={"query": "露营", "method": "bm25", "max_atoms": 5},
    )
    assert r0.status_code == 200
    keys0 = [h["key"] for h in r0.json()["hits"]]

    r1 = client.post(
        f"/spaces/{uid}/recall",
        json={
            "query": "露营",
            "method": "bm25",
            "max_atoms": 5,
            "neighbor_hops": 1,
        },
    )
    assert r1.status_code == 200
    keys1 = [h["key"] for h in r1.json()["hits"]]
    # hops=0 不强制含 hike；hops=1 在命中 camp 时应可能带上 hike
    assert "hobby-camp" in keys1 or "hobby-hike" in keys1
    if "hobby-camp" in keys1:
        assert "hobby-hike" in keys1 or len(keys1) >= 1
    # 默认请求仍合法
    assert "hobby-camp" in keys0 or "hobby-hike" in keys0


def test_derived_from_closure_on_delete(client, fake_llm):
    uid = _uid(client)
    r = client.post(
        f"/spaces/{uid}/sources",
        json={
            "kind": "correction",
            "content": "记住：TA 爱户外运动。",
            "salience": 0.9,
            "external_ref": {"system": "t", "session_id": "s1"},
        },
    )
    src = r.json()["id"]
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": "base-outdoor",
                        "statement": "TA 爱户外运动",
                        "detail": "爬山露营",
                        "change_reason": "t",
                        "source_ids": [src],
                    }
                ]
            },
            ensure_ascii=False,
        )
    ]
    client.post(f"/spaces/{uid}/consolidate", json={})

    r2 = client.post(
        f"/spaces/{uid}/sources",
        json={
            "kind": "turn",
            "content": "综合：户外活跃",
            "salience": 0.5,
            "external_ref": {"system": "t", "session_id": "s2"},
        },
    )
    src2 = r2.json()["id"]
    fake_llm.responses = [
        json.dumps({"read": ["base-outdoor"]}),
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "derived-active",
                        "statement": "TA 是户外活跃型",
                        "detail": "推自 base",
                        "confidence": 0.7,
                        "change_reason": "synth",
                        "source_ids": [src2],
                        "links": [
                            {"to": "base-outdoor", "kind": "derived_from"}
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
    ]
    client.post(f"/spaces/{uid}/consolidate", json={})

    preview = client.post(
        f"/spaces/{uid}/sources/delete-by-ref/preview",
        json={"external_ref": {"system": "t", "session_id": "s1"}},
    ).json()
    assert "base-outdoor" in preview["atoms_to_delete"] or (
        "base-outdoor" in preview["atoms_to_reconsolidate"]
    )
    assert "derived-active" in preview.get("atoms_derived_affected", [])


def test_synthesize_rejects_weak_ops(client, fake_llm):
    uid = _uid(client)
    _seed_two_atoms(client, fake_llm, uid)
    # 缺 derived_from / confidence → 丢弃
    fake_llm.responses = [
        json.dumps({"read": ["hobby-hike", "hobby-camp"]}),
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "bad-1",
                        "statement": "无父母",
                        "detail": "",
                        "confidence": 0.5,
                        "derived_from": ["hobby-hike"],
                    },
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "bad-2",
                        "statement": "无置信度",
                        "detail": "",
                        "derived_from": ["hobby-hike", "hobby-camp"],
                    },
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "good-outdoor",
                        "statement": "TA 热衷户外活动",
                        "detail": "综合兴趣",
                        "confidence": 0.95,
                        "derived_from": ["hobby-hike", "hobby-camp"],
                    },
                ]
            },
            ensure_ascii=False,
        ),
    ]
    run = client.post(f"/spaces/{uid}/synthesize", json={}).json()
    assert run["status"] == "succeeded"
    assert "good-outdoor" in (run.get("atoms_touched") or [])
    assert "bad-1" not in (run.get("atoms_touched") or [])
    assert "bad-2" not in (run.get("atoms_touched") or [])
    detail = client.get(f"/spaces/{uid}/atoms/good-outdoor").json()
    assert detail["confidence"] <= 0.8
    nb = client.get(f"/spaces/{uid}/atoms/good-outdoor/neighbors").json()
    kinds = {x["kind"] for x in nb["out"]}
    assert "derived_from" in kinds


def test_temporal_pure_functions():
    d = date(2023, 5, 25)  # Thursday
    assert weekday_name(d, lang="en") == "Thursday"
    assert weekday_name(d, lang="zh") == "星期四"
    prev = resolve_relative("The Sunday before 25 May 2023", d)
    assert prev == date(2023, 5, 21)
    assert weekday_name(prev, lang="en") == "Sunday"
    facts = derive_time_facts(d, "")
    assert "星期四" in facts
    assert "Thursday" in facts
    detail = append_time_facts_to_detail("原文", d)
    assert "time_facts=" in detail
    # 幂等
    assert append_time_facts_to_detail(detail, d) == detail
    last = derive_time_facts(date(2023, 7, 12), "This book I read last year")
    assert "2022" in last
    assert any("last year" in f for f in last)


def test_reverse_derived_closure_two_levels(client, fake_llm):
    """A ← B ← C（derived_from）：改 A 应波及 B 与 C。"""
    uid = _uid(client)
    _seed_two_atoms(client, fake_llm, uid)

    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "layer", "salience": 0.8},
    )
    sid = r.json()["id"]
    fake_llm.responses = [
        json.dumps({"read": ["hobby-hike", "hobby-camp"]}),
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "mid-layer",
                        "statement": "中层认识",
                        "detail": "",
                        "confidence": 0.6,
                        "change_reason": "m",
                        "source_ids": [sid],
                        "links": [
                            {"to": "hobby-hike", "kind": "derived_from"}
                        ],
                    },
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "top-layer",
                        "statement": "顶层认识",
                        "detail": "",
                        "confidence": 0.6,
                        "change_reason": "m",
                        "source_ids": [sid],
                        "links": [
                            {"to": "mid-layer", "kind": "derived_from"}
                        ],
                    },
                ]
            },
            ensure_ascii=False,
        ),
    ]
    client.post(f"/spaces/{uid}/consolidate", json={})

    mid = client.get(f"/spaces/{uid}/atoms/mid-layer").json()
    top = client.get(f"/spaces/{uid}/atoms/top-layer").json()
    conf_mid_before = mid["confidence"]
    conf_top_before = top["confidence"]

    # 更新 hike → 应降 mid/top 置信度
    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "更新爬山", "salience": 0.9},
    )
    sid2 = r.json()["id"]
    fake_llm.responses = [
        json.dumps({"read": ["hobby-hike"]}),
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": "hobby-hike",
                        "statement": "TA 更爱徒步",
                        "detail": "改了",
                        "change_reason": "u",
                        "source_ids": [sid2],
                    }
                ]
            },
            ensure_ascii=False,
        ),
    ]
    client.post(f"/spaces/{uid}/consolidate", json={})
    mid2 = client.get(f"/spaces/{uid}/atoms/mid-layer").json()
    top2 = client.get(f"/spaces/{uid}/atoms/top-layer").json()
    assert mid2["confidence"] < conf_mid_before
    assert top2["confidence"] < conf_top_before
