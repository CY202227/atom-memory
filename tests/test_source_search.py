import json


def space(client, name):
    return client.post("/spaces", json={"owner_id": name, "subject_id": "test"}).json()["uid"]


def test_source_lookup_finds_unconsolidated_fact_and_keeps_anchor(client):
    uid = space(client, "reader")
    source = client.post(f"/spaces/{uid}/sources", json={"kind": "turn", "content": "[session_time] 8 May 2023\nMelanie: I painted a sunrise last year.", "salience": 0.2}).json()
    result = client.post(f"/spaces/{uid}/sources/search", json={"query": "When did Melanie paint a sunrise?", "budget_chars": 100}).json()
    assert result["evidence_status"] == "historical_sources"
    assert result["hits"][0]["source_id"] == source["id"]
    assert "8 May 2023" in result["hits"][0]["excerpt"]
    assert "last year" in result["hits"][0]["excerpt"]
    assert result["chars_used"] <= 100


def test_space_isolation_and_hard_deletion(client):
    uid, other = space(client, "owner"), space(client, "other")
    client.post(f"/spaces/{uid}/sources", json={"kind": "manual", "content": "A private astronomy hobby.", "external_ref": {"conversation": "deleted"}})
    payload = {"query": "astronomy"}
    assert client.post(f"/spaces/{other}/sources/search", json=payload).json()["hits"] == []
    assert client.post(f"/spaces/{uid}/sources/search", json=payload).json()["hits"]
    deletion = client.post(f"/spaces/{uid}/sources/delete-by-ref", json={"external_ref": {"conversation": "deleted"}})
    assert deletion.status_code == 200
    assert client.post(f"/spaces/{uid}/sources/search", json=payload).json()["hits"] == []


def test_source_time_does_not_pollute_unrelated_atoms(client, fake_llm):
    uid = space(client, "time")
    source = client.post(f"/spaces/{uid}/sources", json={"kind": "turn", "content": "Alice painted last year. Bob likes tea."}).json()["id"]
    fake_llm.responses = [json.dumps({"operations": [{"op": "upsert", "kind": "person", "key": "bob-tea", "statement": "Bob likes tea.", "detail": "", "source_ids": [source]}]})]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == "succeeded"
    atom = client.get(f"/spaces/{uid}/atoms").json()["results"][0]
    assert "last year" not in atom["detail"]
    assert "year=" not in atom["detail"]


def test_long_paragraph_match_after_first_chunk(client):
    uid = space(client, "long")
    client.post(f"/spaces/{uid}/sources", json={"kind": "manual", "content": "ordinary text " * 100 + "Melanie painted a sunrise."})
    hits = client.post(f"/spaces/{uid}/sources/search", json={"query": "sunrise"}).json()["hits"]
    assert "sunrise" in hits[0]["excerpt"]


def test_matching_passage_uses_its_preceding_session_anchor(client):
    uid = space(client, "anchors")
    client.post(f"/spaces/{uid}/sources", json={"kind": "manual", "content": "[session_time] 2023-05-08\nBob visited a park.\n[session_time] 2024-05-08\nAlice painted a sunrise last year."})
    hit = client.post(f"/spaces/{uid}/sources/search", json={"query": "sunrise"}).json()["hits"][0]
    assert "[session_time] 2024-05-08" in hit["excerpt"]
    assert "year=2023" in hit["excerpt"]
