import pytest
from atom_memory.models import Atom, AtomKind
from atom_memory.recall.llm import LlmRecall, RecallError


def test_semantic_recall_does_not_accept_invented_keys(fake_llm):
    fake_llm.responses = ['{"keys": ["invented"]}']
    with pytest.raises(RecallError, match="unknown memory key"):
        LlmRecall(fake_llm).retrieve([Atom(space_id=1, key="tea", kind=AtomKind.person, statement="Alice likes tea.")], "preferences", 5)


def test_duplicate_keys_do_not_consume_recall_slots(fake_llm):
    fake_llm.responses = ['{"keys": ["tea", "tea", "coffee"]}']
    atoms = [Atom(space_id=1, key=key, kind=AtomKind.person, statement=key) for key in ("tea", "coffee")]
    assert [hit.atom.key for hit in LlmRecall(fake_llm).retrieve(atoms, "drinks", 2).hits] == ["tea", "coffee"]


def test_unrelated_l2_atom_is_not_used_as_recent_fallback():
    from atom_memory.recall.layered import compose_layered
    atom = Atom(space_id=1, key="museum", kind=AtomKind.belief, statement="Alice enjoys museums.", memory_layer=2)
    assert compose_layered([atom], "quantum mechanics", l1_hits=[], max_atoms=5) == []
