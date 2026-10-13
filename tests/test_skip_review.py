import json
from atom_memory.config import settings


def setup_source(client, monkeypatch):
    monkeypatch.setattr(settings, "consolidate_review_skips", True)
    uid = client.post("/spaces", json={"owner_id": "review", "subject_id": "test"}).json()["uid"]
    source = client.post(f"/spaces/{uid}/sources", json={"kind": "turn", "content": "Melanie painted a sunrise last year.", "salience": 0.2}).json()["id"]
    return uid, source


def test_skip_review_recovers_explicit_fact_before_source_is_skipped(client, fake_llm, monkeypatch):
    uid, source = setup_source(client, monkeypatch)
    fake_llm.responses = ['{"operations": []}', json.dumps({"missing": [{"source_id": source, "quote": "painted a sunrise last year"}]}),
        json.dumps({"operations": [{"op": "upsert", "kind": "event", "key": "sunrise", "statement": "Melanie painted a sunrise last year.", "source_ids": [source]}]})]
    run = client.post(f"/spaces/{uid}/consolidate", json={}).json()
    assert run["status"] == "succeeded"
    assert run["atoms_touched"] == ["sunrise"]
    assert client.get(f"/spaces/{uid}/sources").json()[0]["status"] == "consolidated"


def test_review_cannot_use_a_quote_that_was_not_in_the_source(client, fake_llm, monkeypatch):
    uid, source = setup_source(client, monkeypatch)
    invalid = json.dumps({"missing": [{"source_id": source, "quote": "She painted it in 1990"}]})
    fake_llm.responses = ['{"operations": []}', invalid, invalid]
    run = client.post(f"/spaces/{uid}/consolidate", json={}).json()
    assert run["status"] == "failed"
    assert client.get(f"/spaces/{uid}/sources").json()[0]["status"] == "pending"
    assert client.get(f"/spaces/{uid}/atoms").json()["count"] == 0


def test_repair_that_still_omits_reviewed_source_is_not_success(client, fake_llm, monkeypatch):
    uid, source = setup_source(client, monkeypatch)
    fake_llm.responses = ['{"operations": []}', json.dumps({"missing": [{"source_id": source, "quote": "painted a sunrise"}]}), '{"operations": []}', '{"operations": []}']
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == "failed"
    assert client.get(f"/spaces/{uid}/sources").json()[0]["status"] == "pending"


def test_agreed_no_new_facts_can_still_be_skipped(client, fake_llm, monkeypatch):
    uid, _ = setup_source(client, monkeypatch)
    fake_llm.responses = ['{"operations": []}', '{"missing": []}']
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == "succeeded"
    assert client.get(f"/spaces/{uid}/sources").json()[0]["status"] == "skipped"
