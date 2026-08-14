"""评测辅助（LoCoMo / PersonaMem 等）；非运行时依赖。

独立跑分入口::

    python -m atom_memory.eval.run_locomo --mini
    python -m atom_memory.eval.run_personamem --mini
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
from .personamem import (
    PersonaMemBenchReport,
    PersonaMemQAItem,
    PersonaMemReport,
    PersonaMemSample,
    aggregate_reports as aggregate_personamem_reports,
    evaluate_personamem_sample,
    format_personamem_scoreboard,
    ingest_personamem_sample,
    load_personamem,
    oracle_write_plan_personamem,
    parse_mcq_option,
)

__all__ = [
    "LocomoBenchReport",
    "LocomoQAItem",
    "LocomoReport",
    "LocomoSample",
    "PersonaMemBenchReport",
    "PersonaMemQAItem",
    "PersonaMemReport",
    "PersonaMemSample",
    "aggregate_personamem_reports",
    "aggregate_reports",
    "evaluate_locomo_sample",
    "evaluate_personamem_sample",
    "format_personamem_scoreboard",
    "format_scoreboard",
    "ingest_locomo_sample",
    "ingest_personamem_sample",
    "load_locomo",
    "load_personamem",
    "locomo_dia_key",
    "oracle_write_plan",
    "oracle_write_plan_personamem",
    "parse_mcq_option",
    "parse_session_date",
    "score_answer_in_text",
    "score_evidence_keys",
    "token_f1",
]
