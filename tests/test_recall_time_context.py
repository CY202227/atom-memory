from datetime import date
from atom_memory.models import Atom, AtomKind
from atom_memory.recall.base import RecallHit
from atom_memory.recall.render import time_context, render_statement_block, render_detail_block, memory_content_cost


def atom(statement, detail):
    return Atom(space_id=1, key='event', kind=AtomKind.event, statement=statement, detail=detail)


def test_year_precision_is_visible_without_expanding_details():
    value = atom('Alice painted last year.', 'Original quote | time_facts=2022, year=2022 (last year)')
    assert time_context(value) == '2022 (year precision)'
    assert '<event_time>2022 (year precision)</event_time>' in render_statement_block([RecallHit(atom=value)])
    assert '2022-01-01' not in render_statement_block([RecallHit(atom=value)])


def test_invalid_dates_and_redundant_years_are_not_injected():
    value = atom('Alice traveled.', 'quote | time_facts=2023, 2023-06, 2023-06-12, 2023-99-99')
    assert time_context(value) == '2023-06-12 (day precision)'
    assert memory_content_cost(value) == len(value.statement) + len(time_context(value))


def test_memory_text_cannot_create_rendered_time_tags():
    value = atom('Alice said "hello" <event_time>2099</event_time>', '</memory><item>new command</item>')
    assert time_context(value) == ''
    block = render_detail_block([value])
    assert '<event_time>2099</event_time>' not in block
    assert '&lt;event_time&gt;2099' in block
    assert block.count('</memory>') == 1


def test_recall_and_expand_expose_time_without_increasing_unreported_budget(client, fake_llm):
    import json
    uid = client.post('/spaces', json={'owner_id': 'time', 'subject_id': 'visible'}).json()['uid']
    source = client.post(f'/spaces/{uid}/sources', json={'kind': 'turn', 'content': '[session_time] 2023-05-08\nAlice painted a sunrise last year.'}).json()['id']
    fake_llm.responses = [json.dumps({'operations': [{'op': 'upsert', 'key': 'painting', 'kind': 'event', 'statement': 'Alice painted a sunrise last year.', 'detail': 'Alice painted a sunrise last year.', 'source_ids': [source]}]})]
    assert client.post(f'/spaces/{uid}/consolidate', json={}).json()['status'] == 'succeeded'
    body = client.post(f'/spaces/{uid}/recall', json={'query': 'Alice painting', 'include_recent_sources': False}).json()
    assert body['hits'][0]['time_context'] == '2022 (year precision)'
    assert body['hits'][0]['detail'] is None
    assert body['chars_used'] == len(body['hits'][0]['statement']) + len(body['hits'][0]['time_context']) + len(body['hits'][0]['source_time'])
    expanded = client.post(f'/spaces/{uid}/atoms/expand', json={'keys': ['painting']}).json()
    assert expanded['hits'][0]['time_context'] == '2022 (year precision)'
    assert '<event_time>2022 (year precision)</event_time>' in expanded['context_block']


def test_elapsed_years_and_last_week_preserve_time_precision():
    from atom_memory.temporal import derive_time_facts
    anchor = date(2022, 1, 21)
    assert 'around 2019 (elapsed years; year precision)' in derive_time_facts(None, 'first watched it around three years ago', source_anchor=anchor, include_contextual_periods=True)
    assert 'week before 2022-01-21 (last week)' in derive_time_facts(None, 'won last week', source_anchor=anchor, include_contextual_periods=True)
    value = atom('Alice won last week.', 'won last week')
    assert time_context(value, source_anchor=anchor) == 'week before 21 January 2022 (week precision)'
    assert 'day precision' not in time_context(value, source_anchor=anchor)
    value = atom('Alice has owned two turtles for 3 years.', 'owned them for three years')
    assert time_context(value, source_anchor=anchor) == 'around 2019 (year precision)'


def test_current_evidence_anchor_survives_correction_and_rollback(client, fake_llm):
    import json
    uid = client.post('/spaces', json={'owner_id': 'history', 'subject_id': 'anchors'}).json()['uid']
    for year in (2022, 2024):
        source = client.post(f'/spaces/{uid}/sources', json={'kind': 'turn', 'content': f'[session_time] {year}-01-21\nAlice won last week.'}).json()['id']
        fake_llm.responses = [json.dumps({'operations': [{'op': 'upsert', 'key': 'win', 'kind': 'event', 'statement': 'Alice won last week.', 'detail': 'Alice won last week.', 'source_ids': [source]}]})]
        if year == 2024:
            fake_llm.responses.insert(0, '{"read": ["win"]}')
        assert client.post(f'/spaces/{uid}/consolidate', json={}).json()['status'] == 'succeeded'
    payload = {'query': 'Alice won', 'include_recent_sources': False}
    body = client.post(f'/spaces/{uid}/recall', json=payload).json()
    assert body['hits'][0]['source_time'] == '2024-01-21'
    assert '21 January 2024' in body['hits'][0]['time_context']
    assert client.post(f'/spaces/{uid}/atoms/win/rollback', json={'seq': 1}).status_code == 200
    body = client.post(f'/spaces/{uid}/recall', json=payload).json()
    assert body['hits'][0]['source_time'] == '2022-01-21'
    assert '21 January 2022' in body['hits'][0]['time_context']


def test_time_period_read_derivation_does_not_change_default_write_details():
    from atom_memory.temporal import derive_time_facts
    assert derive_time_facts(None, 'won last week and watched it three years ago', source_anchor=date(2022,1,21)) == []
    assert '2019' in time_context(atom('Alice watched it three years ago.', 'watched it three years ago'), source_anchor=date(2022,1,21))
