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
    assert body["candidate_count"] == 64
    assert body["candidate_chars"] <= 24000
    assert body["total_passages"] > body["candidate_count"]
    assert len(fake_llm.calls[-1][1]) < 24500


def test_two_passages_in_one_source_obey_total_budget(client, fake_llm):
    uid = create_space(client, "two-facts")
    source = client.post(f"/spaces/{uid}/sources", json={"kind": "manual", "content": "Alice loves birds.\nAlice enjoys hiking."}).json()["id"]
    fake_llm.responses = [json.dumps({"queries": ["Alice hobbies"]}), json.dumps({"keys": [f"source-{source}-0-0", f"source-{source}-1-0"]})]
    body = client.post(f"/spaces/{uid}/sources/search", json={"query": "hobbies", "method": "llm", "budget_chars": 40}).json()
    assert body["chars_used"] == sum(len(hit["excerpt"]) for hit in body["hits"])
    assert body["chars_used"] <= 40


def test_distinct_sources_keep_a_share_of_the_evidence_budget(client, fake_llm):
    uid = create_space(client, 'aggregation')
    ids = []
    keys = []
    for index in range(4):
        source = client.post(f'/spaces/{uid}/sources', json={'kind': 'manual', 'content': 'Context ' * 100 + f'\nAlice completed project {index}.\n' + 'More context ' * 100}).json()['id']
        ids.append(source)
        keys.append(f'source-{source}-1-0')
    fake_llm.responses = [json.dumps({'queries': ['Alice projects']}), json.dumps({'keys': keys})]
    body = client.post(f'/spaces/{uid}/sources/search', json={'query': 'List Alice projects', 'method': 'llm', 'max_sources': 4, 'budget_chars': 1200}).json()
    assert [hit['source_id'] for hit in body['hits']] == ids
    assert all(f'Alice completed project {index}.' in hit['excerpt'] for index, hit in enumerate(body['hits']))
    assert body['chars_used'] <= 1200


def test_second_selected_fact_precedes_long_neighbor_context(client, fake_llm):
    uid = create_space(client, 'two-independent-facts')
    source = client.post(f'/spaces/{uid}/sources', json={'kind': 'manual', 'content': 'Alice likes birds.\n' + 'Unrelated context ' * 60 + '\nAlice likes hiking.'}).json()['id']
    fake_llm.responses = [json.dumps({'queries': ['hobbies']}), json.dumps({'keys': [f'source-{source}-0-0', f'source-{source}-2-0']})]
    body = client.post(f'/spaces/{uid}/sources/search', json={'query': 'hobbies', 'method': 'llm', 'budget_chars': 100}).json()
    assert 'Alice likes birds.' in body['hits'][0]['excerpt']
    assert 'Alice likes hiking.' in body['hits'][0]['excerpt']
    assert body['chars_used'] <= 100


def test_each_selected_fact_keeps_its_own_historical_anchor(client, fake_llm):
    uid = create_space(client, 'multi-anchor')
    source = client.post(f'/spaces/{uid}/sources', json={'kind': 'manual', 'content': '[session_time] 2023-05-08\nAlice painted last year.\n[session_time] 2024-05-08\nAlice traveled last year.'}).json()['id']
    fake_llm.responses = [json.dumps({'queries': ['Alice activities']}), json.dumps({'keys': [f'source-{source}-1-0', f'source-{source}-3-0']})]
    excerpt = client.post(f'/spaces/{uid}/sources/search', json={'query': 'activities', 'method': 'llm'}).json()['hits'][0]['excerpt']
    assert excerpt.index('2023-05-08') < excerpt.index('Alice painted') < excerpt.index('2024-05-08') < excerpt.index('Alice traveled')
