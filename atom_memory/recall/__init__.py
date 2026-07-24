"""召回策略包。"""

from .base import RecallHit, RecallOutcome, RecallStrategy
from .bm25 import Bm25Recall
from .fuzzy import FuzzyRecall
from .llm import LlmRecall
from .render import (
    clip_by_budget,
    merge_context,
    render_detail_block,
    render_recent_sources,
    render_statement_block,
    truncate_source,
)

METHODS = ("fuzzy", "bm25", "llm")

__all__ = [
    "RecallHit",
    "RecallOutcome",
    "RecallStrategy",
    "FuzzyRecall",
    "Bm25Recall",
    "LlmRecall",
    "render_statement_block",
    "render_detail_block",
    "clip_by_budget",
    "merge_context",
    "render_recent_sources",
    "truncate_source",
    "METHODS",
]
