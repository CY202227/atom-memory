import json


def test_retry_is_idempotent_and_conflicts_are_rejected(client, fake_llm):
    uid = "retry-space"
    client.post("/spaces", json={"uid": uid})
    body = {"kind": "correction", "content": "请叫我小周", "salience": 0.9, "idempotency_key": "turn-1"}
    first = client.post(f"/spaces/{uid}/sources", json=body)
    assert first.status_code == 200
    source_id = first.json()["id"]
    assert client.post(f"/spaces/{uid}/sources", json=body).json()["id"] == source_id
    assert len(client.get(f"/spaces/{uid}/sources").json()) == 1
    assert client.post(f"/spaces/{uid}/sources", json={**body, "content": "叫我小李"}).status_code == 409
    fake_llm.responses.append(json.dumps({"operations": []}))
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == "succeeded"
    assert client.get(f"/spaces/{uid}/sources/{source_id}").json()["status"] == "pending"
    assert client.post(f"/spaces/{uid}/sources", json=body).json()["id"] == source_id
    assert client.get("/spaces/other-space/sources/" + str(source_id)).status_code == 404


def test_without_key_allows_repeated_experiences(client):
    uid = "repeat-space"
    client.post("/spaces", json={"uid": uid})
    body = {"kind": "turn", "content": "今天跑步了"}
    a = client.post(f"/spaces/{uid}/sources", json=body).json()
    b = client.post(f"/spaces/{uid}/sources", json=body).json()
    assert a["id"] != b["id"]


def test_existing_database_upgrade(monkeypatch, tmp_path):
    from sqlalchemy import inspect, text
    from sqlmodel import create_engine
    from atom_memory import db
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE source (id INTEGER PRIMARY KEY, space_id INTEGER NOT NULL)"))
        conn.execute(text("INSERT INTO source (id, space_id) VALUES (1, 7)"))
    monkeypatch.setattr(db, "engine", engine)
    db._ensure_source_idempotency()
    db._ensure_source_idempotency()
    with engine.begin() as conn:
        assert "idempotency_key" in {c["name"] for c in inspect(conn).get_columns("source")}
        assert conn.execute(text("SELECT id FROM source")).scalar() == 1
        conn.execute(text("UPDATE source SET idempotency_key='retry' WHERE id=1"))
        conn.execute(text("INSERT INTO source (id, space_id, idempotency_key) VALUES (2, 8, 'retry')"))
    from sqlalchemy.exc import IntegrityError
    import pytest
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(text("INSERT INTO source (id, space_id, idempotency_key) VALUES (3, 7, 'retry')"))
    engine.dispose()


def test_retry_timestamp_compares_instants(client):
    uid = "timestamp-space"
    client.post("/spaces", json={"uid": uid})
    body = {"kind": "turn", "content": "跑步", "idempotency_key": "time-1", "occurred_at": "2026-10-10T17:00:00+08:00"}
    first = client.post(f"/spaces/{uid}/sources", json=body)
    retry = client.post(f"/spaces/{uid}/sources", json={**body, "occurred_at": "2026-10-10T09:00:00Z"})
    assert retry.status_code == 200
    assert retry.json()["id"] == first.json()["id"]
