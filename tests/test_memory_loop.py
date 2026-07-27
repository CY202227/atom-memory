"""端到端 Atom-first：ingest → 固化 → 召回 → 回滚 → 归档。"""

import json


def _create_space(client, **kw):
    r = client.post("/spaces", json=kw or {"owner_id": "u1", "subject_id": "role1"})
    assert r.status_code == 200
    return r.json()["uid"]


def test_full_loop(client, fake_llm):
    uid = _create_space(client)

    r = client.post(
        f"/spaces/{uid}/sources",
        json={
            "kind": "correction",
            "content": "别再叫我张先生，叫我老张。上次任务里你把日报发成了周报。",
            "salience": 0.9,
        },
    )
    src1 = r.json()["id"]
    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "用户：早上好。AI：早上好呀。"},
    )
    src2 = r.json()["id"]

    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "lesson",
                        "key": "report-format-mistake",
                        "statement": "给老张交日报须按天维度，勿用周报格式",
                        "detail": "先确认报告周期再输出。",
                        "change_reason": "用户指出错误",
                        "source_ids": [src1],
                    },
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": "lao-zhang",
                        "statement": "称呼偏好老张，勿称张先生",
                        "detail": "",
                        "change_reason": "用户纠正称呼",
                        "source_ids": [src1],
                    },
                ]
            },
            ensure_ascii=False,
        )
    ]
    r = client.post(f"/spaces/{uid}/consolidate", json={})
    run = r.json()
    assert run["status"] == "succeeded"
    assert sorted(run["atoms_touched"]) == ["lao-zhang", "report-format-mistake"]
    assert "pages_touched" not in run
    assert len(fake_llm.calls) == 1

    sources = {s["id"]: s for s in client.get(f"/spaces/{uid}/sources").json()}
    assert sources[src1]["status"] == "consolidated"
    assert sources[src2]["status"] == "skipped"

    index = client.get(f"/spaces/{uid}/index").json()
    assert {e["key"] for e in index} == {"lao-zhang", "report-format-mistake"}
    assert all("hook" not in e and "slug" not in e for e in index)

    ev = client.get(f"/spaces/{uid}/atoms/report-format-mistake/evidence").json()
    assert ev[0]["source_id"] == src1 and ev[0]["revision_seq"] == 1

    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "diary", "content": "今天老张说他每周五要一份周总结。", "salience": 0.5},
    )
    src3 = r.json()["id"]
    fake_llm.responses = [
        json.dumps({"read": ["lao-zhang"]}),
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": "lao-zhang",
                        "statement": "称呼老张；每周五要周总结",
                        "detail": "周五提供周总结。",
                        "change_reason": "新增约定",
                        "source_ids": [src3],
                    }
                ]
            },
            ensure_ascii=False,
        ),
    ]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == "succeeded"
    revs = client.get(f"/spaces/{uid}/atoms/lao-zhang/revisions").json()
    assert [rv["seq"] for rv in revs] == [1, 2]
    assert "周总结" in client.get(f"/spaces/{uid}/atoms/lao-zhang").json()["statement"]

    r = client.post(f"/spaces/{uid}/atoms/lao-zhang/rollback", json={"seq": 1})
    assert "周总结" not in r.json()["statement"]
    revs = client.get(f"/spaces/{uid}/atoms/lao-zhang/revisions").json()
    assert [rv["seq"] for rv in revs] == [1, 2, 3]
    assert revs[2]["trigger"] == "rollback"

    fake_llm.responses = [json.dumps({"keys": ["report-format-mistake"]})]
    r = client.post(
        f"/spaces/{uid}/recall",
        json={"query": "又要给老张写报告了，注意什么？", "method": "llm"},
    )
    body = r.json()
    assert [h["key"] for h in body["hits"]] == ["report-format-mistake"]
    assert "hook" not in body["hits"][0] and "type" not in body["hits"][0]
    assert "<recalled_memory>" in body["context_block"]
    assert "日报" in body["context_block"]


def test_keyword_recall_and_budget(client, fake_llm):
    uid = _create_space(client, owner_id="u9", subject_id="r9")
    client.post(f"/spaces/{uid}/sources", json={"kind": "manual", "content": "材料"})
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "lesson",
                        "key": "report-mistake",
                        "statement": "日报误用周报格式，须先确认周期",
                        "detail": "按天输出。",
                        "change_reason": "建",
                        "source_ids": [1],
                    },
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "coffee-preference",
                        "statement": "用户喜欢冰美式咖啡",
                        "detail": "",
                        "change_reason": "建",
                        "source_ids": [1],
                    },
                ]
            },
            ensure_ascii=False,
        )
    ]
    client.post(f"/spaces/{uid}/consolidate", json={})
    fake_llm.calls.clear()

    for method in ("bm25", "fuzzy"):
        r = client.post(
            f"/spaces/{uid}/recall",
            json={"query": "写日报要注意什么", "method": method, "include_recent_sources": False},
        )
        body = r.json()
        assert body["hits"][0]["key"] == "report-mistake", method
        assert "日报" in body["context_block"]

    r = client.post(
        f"/spaces/{uid}/recall",
        json={
            "query": "写日报要注意什么",
            "method": "bm25",
            "budget_chars": 40,
            "include_recent_sources": False,
        },
    )
    assert r.status_code == 200
    assert len(r.json()["hits"]) == 1

    r = client.post(
        f"/spaces/{uid}/recall",
        json={"query": "量子力学", "method": "bm25", "include_recent_sources": False},
    )
    assert r.json()["hits"] == [] and r.json()["context_block"] == ""
    assert fake_llm.calls == []


def test_include_recent_sources_before_consolidate(client, fake_llm):
    uid = _create_space(client, owner_id="u-pending", subject_id="r-pending")
    client.post(
        f"/spaces/{uid}/sources",
        json={
            "kind": "correction",
            "content": "请记住我叫小美，不要叫错名字",
            "salience": 0.9,
        },
    )
    r = client.post(
        f"/spaces/{uid}/recall",
        json={"query": "我叫什么", "method": "bm25", "include_recent_sources": True},
    )
    body = r.json()
    assert body["hits"] == []
    assert "recent_sources" in body["context_block"]
    assert "小美" in body["context_block"]


def test_space_uid_identity(client, fake_llm):
    r = client.post("/spaces", json={"uid": "my-app-uuid-0001"})
    assert r.json()["uid"] == "my-app-uuid-0001"
    r2 = client.post("/spaces", json={"uid": "my-app-uuid-0001"})
    assert r2.json()["id"] == r.json()["id"]

    a = client.post("/spaces", json={"owner_id": "u1", "subject_id": "r1"}).json()
    b = client.post("/spaces", json={"owner_id": "u1", "subject_id": "r1"}).json()
    assert a["id"] == b["id"]

    c = client.post("/spaces", json={}).json()
    d = client.post("/spaces", json={}).json()
    assert c["uid"] != d["uid"]

    spaces = client.get("/spaces", params={"owner_id": "u1"}).json()
    assert [s["id"] for s in spaces] == [a["id"]]

    client.post(f"/spaces/{a['uid']}/sources", json={"kind": "manual", "content": "A 的记忆材料"})
    assert client.get(f"/spaces/{c['uid']}/sources").json() == []
    assert client.get("/spaces/no-such-uid/index").status_code == 404


def test_consolidate_zero_ops_and_failure(client, fake_llm):
    uid = _create_space(client, owner_id="u3", subject_id="r3")
    r = client.post(f"/spaces/{uid}/consolidate", json={})
    assert r.json()["status"] == "succeeded" and r.json()["atoms_touched"] == []
    assert fake_llm.calls == []

    client.post(f"/spaces/{uid}/sources", json={"kind": "turn", "content": "你好。你好呀。"})
    fake_llm.responses = [json.dumps({"operations": []})]
    r = client.post(f"/spaces/{uid}/consolidate", json={})
    assert r.json()["status"] == "succeeded"
    assert client.get(f"/spaces/{uid}/sources").json()[0]["status"] == "skipped"

    # 高 salience + 空 ops：保持 pending，可重试
    client.post(
        f"/spaces/{uid}/sources",
        json={
            "kind": "correction",
            "content": "请记住：发售日类事实须先联网搜索。",
            "salience": 0.9,
        },
    )
    fake_llm.responses = [json.dumps({"operations": []})]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == "succeeded"
    pending_hi = client.get(f"/spaces/{uid}/sources", params={"status": "pending"}).json()
    assert len(pending_hi) == 1
    assert pending_hi[0]["kind"] == "correction"

    client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "manual", "content": "重要材料", "salience": 1.0},
    )
    fake_llm.responses = ["我不会输出 JSON。"]
    r = client.post(f"/spaces/{uid}/consolidate", json={})
    run = r.json()
    assert run["status"] == "failed" and run["error"]
    pending = client.get(f"/spaces/{uid}/sources", params={"status": "pending"}).json()
    # correction(0.9) + manual(1.0) 都仍 pending
    assert len(pending) == 2


def test_uncited_correction_stays_pending_and_priority(client, fake_llm):
    """低优 turn 被 cite → consolidated；未 cite 的 correction 仍 pending 且优先入批。"""
    uid = _create_space(client, owner_id="u-corr", subject_id="r-corr")
    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "用户：闲聊。AI：嗯。", "salience": 0.1},
    )
    turn_id = r.json()["id"]
    r = client.post(
        f"/spaces/{uid}/sources",
        json={
            "kind": "correction",
            "content": "鸣潮不是新游戏，别再当成新发售。",
            "salience": 0.9,
        },
    )
    corr_id = r.json()["id"]

    # 空索引：仅 write 一阶段；只 cite 低优 turn
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "event",
                        "key": "chitchat",
                        "statement": "曾有一轮无实质闲聊",
                        "detail": "",
                        "change_reason": "仅处理 turn",
                        "source_ids": [turn_id],
                    }
                ]
            },
            ensure_ascii=False,
        )
    ]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == "succeeded"
    by_id = {s["id"]: s for s in client.get(f"/spaces/{uid}/sources").json()}
    assert by_id[turn_id]["status"] == "consolidated"
    assert by_id[corr_id]["status"] == "pending"

    # 有 atom 后：select + write；空 ops → correction 仍 pending
    fake_llm.responses = [
        json.dumps({"read": []}),
        json.dumps({"operations": []}),
    ]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == "succeeded"
    pending = client.get(f"/spaces/{uid}/sources", params={"status": "pending"}).json()
    assert len(pending) == 1 and pending[0]["id"] == corr_id

    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "用户：又闲聊。AI：好。", "salience": 0.05},
    )
    turn2_id = r.json()["id"]

    fake_llm.calls.clear()
    fake_llm.responses = [
        json.dumps({"read": []}),
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "lesson",
                        "key": "no-old-as-new",
                        "statement": "勿把已上线老游当成新发售",
                        "detail": "",
                        "change_reason": "用户纠正",
                        "source_ids": [corr_id],
                    }
                ]
            },
            ensure_ascii=False,
        ),
    ]
    run = client.post(f"/spaces/{uid}/consolidate", json={}).json()
    assert run["status"] == "succeeded"
    assert "no-old-as-new" in run["atoms_touched"]

    # write 阶段 user payload：高 salience correction 应排在低优 turn 之前
    write_user = fake_llm.calls[-1][1]
    assert write_user.find(f"source_id={corr_id}") < write_user.find(
        f"source_id={turn2_id}"
    )

    by_id = {s["id"]: s for s in client.get(f"/spaces/{uid}/sources").json()}
    assert by_id[corr_id]["status"] == "consolidated"
    assert by_id[turn2_id]["status"] == "skipped"


def test_ui_served(client):
    r = client.get("/ui")
    assert r.status_code == 200
    assert "atom-memory" in r.text


def test_archive_and_delete_space(client, fake_llm):
    uid = _create_space(client, owner_id="u5", subject_id="r5")
    client.post(f"/spaces/{uid}/sources", json={"kind": "manual", "content": "材料"})
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "b1",
                        "statement": "某认识内容甲",
                        "detail": "",
                        "change_reason": "建",
                        "source_ids": [1],
                    }
                ]
            },
            ensure_ascii=False,
        )
    ]
    client.post(f"/spaces/{uid}/consolidate", json={})

    r = client.post(f"/spaces/{uid}/atoms/b1/archive")
    assert r.json()["status"] == "archived"
    assert client.get(f"/spaces/{uid}/index").json() == []
    r = client.post(
        f"/spaces/{uid}/recall",
        json={"query": "内容甲 认识", "method": "bm25", "include_recent_sources": False},
    )
    assert r.json()["hits"] == []

    counts = client.delete(f"/spaces/{uid}").json()
    assert counts["atoms"] == 1 and counts["sources"] >= 1
    assert client.get(f"/spaces/{uid}/index").status_code == 404


def test_expand_atoms(client, fake_llm):
    uid = _create_space(client, owner_id="u-exp", subject_id="r-exp")
    client.post(f"/spaces/{uid}/sources", json={"kind": "manual", "content": "材料"})
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": "coriander",
                        "statement": "她讨厌香菜",
                        "detail": "点餐注意避香菜。",
                        "happened_on": "2026-07-01",
                        "change_reason": "建",
                        "source_ids": [1],
                    },
                    {
                        "op": "upsert",
                        "kind": "procedure",
                        "key": "report-habit",
                        "statement": "日报按天写",
                        "detail": "别用周报格式。",
                        "change_reason": "建",
                        "source_ids": [1],
                    },
                ]
            },
            ensure_ascii=False,
        )
    ]
    client.post(f"/spaces/{uid}/consolidate", json={})

    r = client.post(
        f"/spaces/{uid}/atoms/expand",
        json={"keys": ["report-habit", "coriander"]},
    )
    body = r.json()
    assert [h["key"] for h in body["hits"]] == ["report-habit", "coriander"]
    assert "点餐注意" in body["context_block"]

    assert client.post(f"/spaces/{uid}/atoms/coriander/archive").status_code == 200
    r = client.post(
        f"/spaces/{uid}/atoms/expand",
        json={"keys": ["coriander", "no-such", "report-habit"]},
    )
    body = r.json()
    assert [h["key"] for h in body["hits"]] == ["report-habit"]
    assert body["missing"] == ["coriander", "no-such"]


def test_statement_truncation(client, fake_llm):
    uid = _create_space(client, owner_id="u-trunc", subject_id="r-trunc")
    client.post(f"/spaces/{uid}/sources", json={"kind": "manual", "content": "材料"})
    long_s = "钩" * 100
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "long-s",
                        "statement": long_s,
                        "detail": "x" * 500,
                        "change_reason": "建",
                        "source_ids": [1],
                    }
                ]
            },
            ensure_ascii=False,
        )
    ]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == "succeeded"
    page = client.get(f"/spaces/{uid}/atoms/long-s").json()
    assert len(page["statement"]) == 80
    assert len(page["detail"]) == 300


def test_consolidate_mutex_returns_active_run(fake_llm):
    from datetime import timedelta

    from sqlalchemy.pool import StaticPool
    from sqlmodel import Session, SQLModel, create_engine

    from atom_memory.consolidation.engine import ConsolidationEngine
    from atom_memory.models import ConsolidationRun, RunStatus, Space, utcnow

    engine_db = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SQLModel.metadata.create_all(engine_db)
    engine = ConsolidationEngine(fake_llm)
    with Session(engine_db) as session:
        space = Space(owner_id="u6", subject_id="r6")
        session.add(space)
        session.commit()
        session.refresh(space)

        active = ConsolidationRun(space_id=space.id, status=RunStatus.running)
        session.add(active)
        session.commit()
        session.refresh(active)
        got = engine.run(session, space)
        assert got.id == active.id and fake_llm.calls == []

        active.started_at = utcnow() - timedelta(seconds=3600)
        session.add(active)
        session.commit()
        fresh = engine.run(session, space)
        assert fresh.id != active.id and fresh.status == RunStatus.succeeded
