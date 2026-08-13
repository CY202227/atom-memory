"""评测辅助（LoCoMo 等）；非运行时依赖。

独立跑分入口::

    python -m atom_memory.eval.run_locomo --mini
"""

from .locomo import (
    LocomoBenchReport,
    LocomoQAItem,
    LocomoReport,
    LocomoSample,
    aggregate_reports,
    evaluate_locomo_sample,
    format_scoreboard,
    ingest_locomo_sample,
    load_locomo,
    locomo_dia_key,
    oracle_write_plan,
    parse_session_date,
    score_answer_in_text,
    score_evidence_keys,
    token_f1,
)

__all__ = [
    "LocomoBenchReport",
    "LocomoQAItem",
    "LocomoReport",
    "LocomoSample",
    "aggregate_reports",
    "evaluate_locomo_sample",
    "format_scoreboard",
    "ingest_locomo_sample",
    "load_locomo",
    "locomo_dia_key",
    "oracle_write_plan",
    "parse_session_date",
    "score_answer_in_text",
    "score_evidence_keys",
    "token_f1",
]
