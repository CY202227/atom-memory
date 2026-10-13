import json


def create_space(client, owner):
    return client.post("/spaces", json={"owner_id": owner, "subject_id": "semantic"}).json()["uid"]


def test_cross_language_lookup_returns_existing_evidence(client, fake_llm):
    uid, other = create_space(client, "reader"), create_space(client, "other")
    source = client.post(f"/spaces/{uid}/sources", json={"kind": "turn", "content": "[session_time] 8 May 2023\nMelanie painted a sunrise last year."}).json()["id"]
    client.post(f"/spaces/{other}/sources", json={"kind": "manual", "content": "private-other-space"})
    fake_llm.responses = [json.dumps({"queries": ["Melanie sunrise painting"]}), json.dumps({"keys": [f"source-{source}-1-0"]})]
    body = client.post(f"/spaces/{uid}/sources/search", json={"query": "她什么时候画的日出？", "method": "llm"}).json()
    assert body["method"] == "llm"
    assert body["coverage"] == "all_passages"
    assert body["hits"][0]["source_id"] == source
    assert "Melanie painted a sunrise" in body["hits"][0]["excerpt"]
    assert "2022" in body["hits"][0]["excerpt"]
    assert "private-other-space" not in fake_llm.calls[-1][1]
    assert all(fmt["json_schema"]["strict"] for fmt in fake_llm.response_formats)


def test_semantic_lookup_rejects_invented_passage_key(client, fake_llm):
    uid = create_space(client, "bad-key")
    client.post(f"/spaces/{uid}/sources", json={"kind": "manual", "content": "Alice enjoys birdwatching."})
    fake_llm.responses = ['{"queries": ["birdwatching"]}', '{"keys": ["invented"]}']
    assert client.post(f"/spaces/{uid}/sources/search", json={"query": "hobbies", "method": "llm"}).status_code == 502


def test_semantic_candidate_subset_is_explicit_and_bounded(client, fake_llm):
    uid = create_space(client, "large")
    client.post(f"/spaces/{uid}/sources", json={"kind": "manual", "content": "\n".join(f"Entry {i}: " + "ordinary words " * 45 for i in range(90))})
    fake_llm.responses = ['{"queries": ["unmatched topic"]}', '{"keys": []}']
    body = client.post(f"/spaces/{uid}/sources/search", json={"query": "unmatched topic", "method": "llm"}).json()
    assert body["coverage"] == "candidate_subset"
    assert body["candidate_count"] <= 64
    assert body["total_passages"] > body["candidate_count"]
    assert len(fake_llm.calls[-1][1]) < 24500


def test_two_passages_in_one_source_obey_total_budget(client, fake_llm):
    uid = create_space(client, "two-facts")
    source = client.post(f"/spaces/{uid}/sources", json={"kind": "manual", "content": "Alice loves birds.\nAlice enjoys hiking."}).json()["id"]
    fake_llm.responses = [json.dumps({"queries": ["Alice hobbies"]}), json.dumps({"keys": [f"source-{source}-0-0", f"source-{source}-1-0"]})]
    body = client.post(f"/spaces/{uid}/sources/search", json={"query": "hobbies", "method": "llm", "budget_chars": 40}).json()
    assert body["chars_used"] == sum(len(hit["excerpt"]) for hit in body["hits"])
    assert body["chars_used"] <= 40
