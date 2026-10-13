import json
from atom_memory.config import settings


def setup(client, monkeypatch):
    monkeypatch.setattr(settings, 'consolidate_extract_facts', True)
    uid = client.post('/spaces', json={'owner_id': 'facts', 'subject_id': 'inventory'}).json()['uid']
    source = client.post(f'/spaces/{uid}/sources', json={'kind': 'turn', 'content': 'Alice painted a sunrise. Alice lives in Paris.'}).json()['id']
    inventory = {'facts': [{'source_id': source, 'quote': 'Alice painted a sunrise.'}, {'source_id': source, 'quote': 'Alice lives in Paris.'}]}
    ops = [{'op': 'upsert', 'key': key, 'statement': quote, 'detail': quote, 'source_ids': [source], 'fact_ids': [index]} for index, key, quote in [(1, 'sunrise', 'Alice painted a sunrise.'), (2, 'home', 'Alice lives in Paris.')]]
    return uid, source, inventory, ops


def test_inventory_facts_survive_in_separate_memories(client, fake_llm, monkeypatch):
    uid, source, inventory, ops = setup(client, monkeypatch)
    fake_llm.responses = list(map(json.dumps, [inventory, {'operations': ops}]))
    assert client.post(f'/spaces/{uid}/consolidate', json={}).json()['status'] == 'succeeded'
    assert client.get(f'/spaces/{uid}/atoms').json()['count'] == 2
    assert client.get(f'/spaces/{uid}/sources').json()[0]['status'] == 'consolidated'


def test_missing_fact_is_retried_without_consuming_source(client, fake_llm, monkeypatch):
    uid, source, inventory, ops = setup(client, monkeypatch)
    fake_llm.responses = list(map(json.dumps, [inventory, {'operations': ops[:1]}, {'operations': ops}]))
    assert client.post(f'/spaces/{uid}/consolidate', json={}).json()['status'] == 'succeeded'
    assert 'omitted inventory fact_ids: [2]' in fake_llm.calls[-1][1]


def test_fact_ids_alone_cannot_claim_retention(client, fake_llm, monkeypatch):
    uid, source, inventory, ops = setup(client, monkeypatch)
    ops[1].update(statement='Alice is nice.', detail='Alice is nice.')
    fake_llm.responses = list(map(json.dumps, [inventory, {'operations': ops}, {'operations': ops}]))
    assert client.post(f'/spaces/{uid}/consolidate', json={}).json()['status'] == 'failed'
    assert client.get(f'/spaces/{uid}/atoms').json()['count'] == 0
    assert client.get(f'/spaces/{uid}/sources').json()[0]['status'] == 'pending'


def test_fabricated_inventory_quote_is_rejected(client, fake_llm, monkeypatch):
    uid, source, inventory, ops = setup(client, monkeypatch)
    inventory['facts'][0]['quote'] = 'Alice sailed to Mars.'
    fake_llm.responses = [json.dumps(inventory)] * 2
    assert client.post(f'/spaces/{uid}/consolidate', json={}).json()['status'] == 'failed'
    assert client.get(f'/spaces/{uid}/sources').json()[0]['status'] == 'pending'


def test_conflicting_same_key_writes_cannot_silently_overwrite(client, fake_llm, monkeypatch):
    uid, source, inventory, ops = setup(client, monkeypatch)
    monkeypatch.setattr(settings, 'consolidate_extract_facts', False)
    ops[1]['key'] = ops[0]['key']
    fake_llm.responses = [json.dumps({'operations': ops})] * 2
    assert client.post(f'/spaces/{uid}/consolidate', json={}).json()['status'] == 'failed'
    assert client.get(f'/spaces/{uid}/atoms').json()['count'] == 0
    assert client.get(f'/spaces/{uid}/sources').json()[0]['status'] == 'pending'


def test_canonical_alias_collision_rolls_back_the_entire_batch(client, fake_llm, monkeypatch):
    uid, source, inventory, ops = setup(client, monkeypatch)
    monkeypatch.setattr(settings, 'consolidate_extract_facts', False)
    ops[0].update(key='user-nickname', kind='person')
    ops[1].update(key='user-preferred-name', kind='person')
    fake_llm.responses = [json.dumps({'operations': ops})]
    assert client.post(f'/spaces/{uid}/consolidate', json={}).json()['status'] == 'failed'
    assert client.get(f'/spaces/{uid}/atoms').json()['count'] == 0
    assert client.get(f'/spaces/{uid}/sources').json()[0]['status'] == 'pending'


def test_time_labels_cannot_truncate_retained_original_quote():
    from datetime import date
    from atom_memory.temporal import append_time_facts_to_detail
    detail = 'supporting evidence ' * 14 + 'final retained fact'
    assert len(detail) < 300
    assert append_time_facts_to_detail(detail, date(2023, 5, 8)) == detail
