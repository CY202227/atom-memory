"""canonical key 合并与近重复。"""

from atom_memory.consolidation.canonical import (
    CANONICAL_PERSONA,
    CANONICAL_PREFERRED_NAME,
    apply_canonical_to_op,
    resolve_write_key,
)
from atom_memory.models import Atom, AtomKind, AtomStatus


def _atom(key: str, statement: str, kind: AtomKind = AtomKind.belief) -> Atom:
    return Atom(
        space_id=1,
        kind=kind,
        key=key,
        statement=statement,
        detail="",
        status=AtomStatus.active,
    )


def test_preferred_name_alias_merges():
    key, note = resolve_write_key(
        key="user-preference-name",
        kind=AtomKind.person,
        statement="称呼老王",
        active_atoms=[],
    )
    assert key == CANONICAL_PREFERRED_NAME
    assert note and "merged_into" in note


def test_persona_alias_merges_to_existing():
    existing = [_atom("persona-catgirl", "我是猫娘", AtomKind.self)]
    key, note = resolve_write_key(
        key="identity-definition",
        kind=AtomKind.self,
        statement="我是带喵口癖的猫娘",
        active_atoms=existing,
    )
    assert key == "persona-catgirl"
    assert note == "merged_into:persona-catgirl"


def test_persona_new_defaults_to_persona():
    key, note = resolve_write_key(
        key="persona-catgirl",
        kind=AtomKind.self,
        statement="我是猫娘",
        active_atoms=[],
    )
    assert key == CANONICAL_PERSONA
    assert note == f"merged_into:{CANONICAL_PERSONA}"


def test_similar_wording_is_not_enough_for_destructive_merge():
    existing = [
        _atom(
            "dental-health",
            "老张补牙较多，担心很快需要根管治疗",
            AtomKind.person,
        )
    ]
    key, note = resolve_write_key(
        key="teeth-worry",
        kind=AtomKind.person,
        statement="老张补牙很多，担心很快就要根管",
        active_atoms=existing,
    )
    assert key == "teeth-worry"
    assert note is None


def test_exact_same_kind_fact_can_still_deduplicate():
    key, note = resolve_write_key(key="tea-copy", kind=AtomKind.person, statement="Alice likes tea.",
        active_atoms=[_atom("alice-tea", "Alice likes tea.", AtomKind.person)])
    assert key == "alice-tea"
    assert note == "merged_into:alice-tea"


def test_third_party_nickname_is_not_primary_user_name():
    key, note = resolve_write_key(key="caroline-nickname", kind=AtomKind.person,
        statement="Caroline is called C.", active_atoms=[])
    assert key == "caroline-nickname"
    assert note is None


def test_equal_short_summary_does_not_erase_different_detail_or_date():
    from datetime import date
    existing = _atom("old-visit", "Alice visited Paris.", AtomKind.event)
    existing.detail = "Visited her brother."
    existing.happened_on = date(2023, 5, 8)
    op, note = apply_canonical_to_op({"op": "upsert", "kind": "event", "key": "new-visit",
        "statement": "Alice visited Paris.", "detail": "Visited a colleague.", "happened_on": "2024-05-08"}, [existing])
    assert op["key"] == "new-visit"
    assert note is None


def test_parallel_preferences_are_retained(client, fake_llm):
    import json
    uid = client.post("/spaces", json={"owner_id": "preferences", "subject_id": "test"}).json()["uid"]
    source = client.post(f"/spaces/{uid}/sources", json={"kind": "turn", "content": "Alice likes red tea and green tea."}).json()["id"]
    fake_llm.responses = [json.dumps({"operations": [
        {"op": "upsert", "kind": "person", "key": "alice-red-tea", "statement": "Alice likes red tea.", "source_ids": [source]},
        {"op": "upsert", "kind": "person", "key": "alice-green-tea", "statement": "Alice likes green tea.", "source_ids": [source]},
    ]})]
    assert client.post(f"/spaces/{uid}/consolidate", json={}).json()["status"] == "succeeded"
    atoms = client.get(f"/spaces/{uid}/atoms").json()
    assert atoms["count"] == 2
    assert {atom["key"] for atom in atoms["results"]} == {"alice-red-tea", "alice-green-tea"}


def test_apply_canonical_updates_op():
    op = {
        "op": "upsert",
        "kind": "person",
        "key": "preferred-name",
        "statement": "称呼老张",
        "change_reason": "纠正",
        "source_ids": [1],
    }
    out, note = apply_canonical_to_op(op, [])
    assert out["key"] == CANONICAL_PREFERRED_NAME
    assert out["kind"] == "person"
    assert note
    assert "merged_into" in out["change_reason"]


def test_distinct_events_are_not_merged_by_similar_wording():
    existing = [Atom(space_id=1, key="paris-trip", kind=AtomKind.event,
        statement="Alice visited Paris in May 2023.")]
    key, note = resolve_write_key(key="rome-trip", kind=AtomKind.event,
        statement="Alice visited Rome in May 2024.", active_atoms=existing)
    assert key == "rome-trip"
    assert note is None


def test_consolidate_forces_canonical_name(client, fake_llm):
    import json

    uid = client.post("/spaces", json={"owner_id": "c1", "subject_id": "r1"}).json()[
        "uid"
    ]
    src = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "correction", "content": "叫我老张", "salience": 0.9},
    ).json()["id"]
    fake_llm.responses = [
        json.dumps(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "kind": "person",
                        "key": "user-preference-name",
                        "statement": "称呼偏好老张",
                        "detail": "",
                        "change_reason": "纠正",
                        "source_ids": [src],
                    }
                ]
            },
            ensure_ascii=False,
        )
    ]
    run = client.post(f"/spaces/{uid}/consolidate", json={}).json()
    assert run["status"] == "succeeded"
    assert run["atoms_touched"] == [CANONICAL_PREFERRED_NAME]
    index = client.get(f"/spaces/{uid}/atoms").json()["results"]
    assert {e["key"] for e in index} == {CANONICAL_PREFERRED_NAME}
