"""delete-by-source：预览分区、执行、原子回退、写锁。"""

import json


def _uid(client, owner="del", subject="s1"):
    return client.post(
        "/spaces", json={"owner_id": owner, "subject_id": subject}
    ).json()["uid"]


def _seed_two_atoms(client, fake_llm, uid):
    """建 session-a-only（仅 A）与 mixed-page（A+B）。"""
    a = client.post(
        f"/spaces/{uid}/sources",
        json={
            "kind": "turn",
            "content": "会话A独有材料",
            "external_ref": {"system": "myapp", "session_id": "s-a"},
            "salience": 0.5,
        },
    ).json()["id"]
    b = client.post(
        f"/spaces/{uid}/sources",
        json={
            "kind": "turn",
            "content": "会话B材料也支撑混合页",
            "external_ref": {"system": "myapp", "session_id": "s-b"},
            "salience": 0.5,
        },
    ).json()["id"]
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "event",
                        "key": "session-a-only",
                        "statement": "仅来自会话A的事件",
                        "detail": "独有细节",
                        "change_reason": "建",
                        "source_ids": [a],
                    },
                    {
                        "op": "upsert",
                        "kind": "belief",
                        "key": "mixed-page",
                        "statement": "混合认识由A与B支撑",
                        "detail": "两边都有",
                        "change_reason": "建",
                        "source_ids": [a, b],
                    },
                ]
            },
            ensure_ascii=False,
        )
    ]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == "succeeded"
    return a, b


def test_preview_partitions_pages(client, fake_llm):
    uid = _uid(client)
    _seed_two_atoms(client, fake_llm, uid)
    r = client.post(
        f"/spaces/{uid}/sources/delete-by-ref/preview",
        json={"external_ref": {"system": "myapp", "session_id": "s-a"}},
    )
    body = r.json()
    assert body["matched_sources"] == 1
    assert body["atoms_to_delete"] == ["session-a-only"]
    assert body["atoms_to_reconsolidate"] == ["mixed-page"]
    assert "pages_to_delete" not in body

    r = client.post(
        f"/spaces/{uid}/sources/delete-by-ref/preview",
        json={"external_ref": {"system": "myapp", "session_id": "nope"}},
    )
    assert r.json()["matched_sources"] == 0

    assert (
        client.post(
            f"/spaces/{uid}/sources/delete-by-ref/preview", json={"external_ref": {}}
        ).status_code
        == 422
    )


def test_execute_deletes_and_reconsolidates(client, fake_llm):
    uid = _uid(client, owner="del2", subject="s2")
    _seed_two_atoms(client, fake_llm, uid)
    fake_llm.responses = [
        json.dumps(
            {
                "statement": "混合认识仅由B支撑",
                "detail": "A已删除",
                "confidence": 0.6,
            },
            ensure_ascii=False,
        )
    ]
    r = client.post(
        f"/spaces/{uid}/sources/delete-by-ref",
        json={"external_ref": {"system": "myapp", "session_id": "s-a"}},
    )
    body = r.json()
    assert body["deleted_sources"] == 1
    assert body["deleted_atoms"] == ["session-a-only"]
    assert body["reconsolidated_atoms"] == ["mixed-page"]
    assert "deleted_pages" not in body
    assert client.get(f"/spaces/{uid}/atoms/session-a-only").status_code == 404
    mixed = client.get(f"/spaces/{uid}/atoms/mixed-page").json()
    assert "仅由B" in mixed["statement"]
    del_run = next(
        x
        for x in client.get(f"/spaces/{uid}/runs").json()
        if x["id"] == body["run_id"]
    )
    assert sorted(del_run["atoms_touched"]) == ["mixed-page", "session-a-only"]


def test_execute_atomic_on_llm_failure(client, fake_llm):
    uid = _uid(client, owner="del3", subject="s3")
    _seed_two_atoms(client, fake_llm, uid)
    fake_llm.responses = ["不是 JSON"]
    r = client.post(
        f"/spaces/{uid}/sources/delete-by-ref",
        json={"external_ref": {"system": "myapp", "session_id": "s-a"}},
    )
    assert r.status_code == 502
    assert client.get(f"/spaces/{uid}/atoms/session-a-only").status_code == 200
    assert len(client.get(f"/spaces/{uid}/sources").json()) == 2


def test_space_write_lock_serializes():
    """同一 space 写锁互斥（delete-by-ref / consolidate 共用）。"""
    import threading
    import time

    from atom_memory.locks import space_write_lock

    overlap = {"max": 0, "cur": 0, "lock": threading.Lock()}

    def work():
        with space_write_lock(42):
            with overlap["lock"]:
                overlap["cur"] += 1
                overlap["max"] = max(overlap["max"], overlap["cur"])
            time.sleep(0.1)
            with overlap["lock"]:
                overlap["cur"] -= 1

    threads = [threading.Thread(target=work) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlap["max"] == 1


def test_execute_no_match_is_noop(client, fake_llm):
    uid = _uid(client, owner="del5", subject="s5")
    r = client.post(
        f"/spaces/{uid}/sources/delete-by-ref",
        json={"external_ref": {"system": "myapp", "session_id": "ghost"}},
    )
    assert r.json() == {
        "deleted_sources": 0,
        "deleted_atoms": [],
        "reconsolidated_atoms": [],
        "run_id": None,
    }
