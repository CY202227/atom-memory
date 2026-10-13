from atom_memory.eval.locomo import score_generated_answer


def test_abstention_containing_gold_is_not_a_correct_answer():
    assert score_generated_answer("Unknown — I cannot confirm whether Caroline is single.", "single") == (0.0, False, True)


def test_explicit_negation_of_gold_is_not_a_hit():
    assert score_generated_answer("She is not single.", "single")[1] is False
    assert score_generated_answer("Not **single**.", "single")[1] is False


def test_gold_word_coverage_does_not_override_low_f1():
    assert score_generated_answer("I cannot know if adoption agencies were researched or just considered.", "adoption agencies")[1] is False


def test_short_answers_and_expected_abstention_are_scored():
    assert score_generated_answer("Adoption agencies", "adoption agencies") == (1.0, True, False)
    assert score_generated_answer("unknown", "unknown") == (1.0, True, True)
    assert score_generated_answer("error: unknown", "unknown")[1] is False
