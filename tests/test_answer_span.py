import pytest
from atom_memory.eval.answer_span import extract_answer_span


def test_normalizer_has_no_gold_and_copies_existing_span(fake_llm):
    fake_llm.responses = ['{"span": "June 2023"}']
    span = extract_answer_span(fake_llm, "When is the trip?", "The trip is planned for June 2023, with the children.")
    assert span == "June 2023"
    assert "gold" not in fake_llm.calls[0][1].lower()


def test_abstention_cannot_be_normalized_into_a_mentioned_alternative(fake_llm):
    assert extract_answer_span(fake_llm, "Relationship?", "Unknown — I cannot confirm whether she is single.") == "unknown"
    assert fake_llm.calls == []
    assert extract_answer_span(fake_llm, "Relationship?", "I don't have a clear record of her relationship status. She plans to adopt as a single parent.") == "unknown"
    assert fake_llm.calls == []


def test_invented_normalized_value_is_rejected(fake_llm):
    fake_llm.responses = ['{"span": "2022"}']
    with pytest.raises(ValueError, match="invented"):
        extract_answer_span(fake_llm, "When?", "She painted it last year.")
