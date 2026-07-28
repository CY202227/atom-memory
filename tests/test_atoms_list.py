"""GET /atoms 分页信封与时间过滤。"""

import json
from datetime import datetime, timedelta, timezone


def _create_space(client, **kw):
    r = client.post("/spaces", json=kw or {"owner_id": "u-list", "subject_id": "r-list"})
    assert r.status_code == 200
    return r.json()["uid"]


def test_atoms_list_pagination_and_updated_after(client, fake_llm):
    uid = _create_space(client)
    client.post(f"/spaces/{uid}/sources", json={"kind": "manual", "content": "材料"})

    ops = [
        {
            "op": "upsert",
            "kind": "person",
            "key": "alice-name",
            "statement": "称呼偏好 Alice，勿称女士",
            "detail": "",
            "change_reason": "建",
            "source_ids": [1],
        },
        {
            "op": "upsert",
            "kind": "lesson",
            "key": "report-format",
            "statement": "日报须按天维度输出，勿用周报模板",
            "detail": "",
            "change_reason": "建",
            "source_ids": [1],
        },
        {
            "op": "upsert",
            "kind": "belief",
            "key": "coffee-pref",
            "statement": "用户长期偏好冰美式而非拿铁",
            "detail": "",
            "change_reason": "建",
            "source_ids": [1],
        },
        {
            "op": "upsert",
            "kind": "event",
            "key": "trip-sf",
            "statement": "计划七月前往旧金山出差一周",
            "detail": "",
            "change_reason": "建",
            "source_ids": [1],
        },
        {
            "op": "upsert",
            "kind": "procedure",
            "key": "deploy-checklist",
            "statement": "发版前必须跑 pytest 并检查迁移脚本",
            "detail": "",
            "change_reason": "建",
            "source_ids": [1],
        },
    ]
    fake_llm.responses = [json.dumps({"operations": ops}, ensure_ascii=False)]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == "succeeded"

    page1 = client.get(
        f"/spaces/{uid}/atoms", params={"page": 1, "page_size": 2}
    ).json()
    assert page1["count"] == 5
    assert page1["page"] == 1
    assert page1["page_size"] == 2
    assert len(page1["results"]) == 2

    page2 = client.get(
        f"/spaces/{uid}/atoms", params={"page": 2, "page_size": 2}
    ).json()
    assert page2["count"] == 5
    assert page2["page"] == 2
    assert len(page2["results"]) == 2
    keys1 = {a["key"] for a in page1["results"]}
    keys2 = {a["key"] for a in page2["results"]}
    assert keys1.isdisjoint(keys2)

    page3 = client.get(
        f"/spaces/{uid}/atoms", params={"page": 3, "page_size": 2}
    ).json()
    assert len(page3["results"]) == 1

    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    empty = client.get(
        f"/spaces/{uid}/atoms", params={"updated_after": future}
    ).json()
    assert empty["count"] == 0
    assert empty["results"] == []

    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    all_after = client.get(
        f"/spaces/{uid}/atoms", params={"updated_after": past, "page_size": 50}
    ).json()
    assert all_after["count"] == 5
    assert len(all_after["results"]) == 5
