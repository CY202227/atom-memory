from atom_memory.models import Atom, AtomKind
from atom_memory.recall.bm25 import Bm25Recall


def test_english_inflections_match_without_handwritten_topic_synonyms():
    atoms = [Atom(space_id=1, key="connections", kind=AtomKind.event,
        statement="Alice was connecting sensors and studying birds.")]
    for query in ("connect sensor", "study bird"):
        assert Bm25Recall().retrieve(atoms, query, 5).hits[0].atom.key == "connections"


def test_chinese_and_numeric_terms_are_preserved():
    atoms = [Atom(space_id=1, key="date", kind=AtomKind.event, statement="老张在2023年参加露营")]
    assert Bm25Recall().retrieve(atoms, "2023 露营", 5).hits[0].atom.key == "date"
