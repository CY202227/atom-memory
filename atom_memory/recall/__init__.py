"""召回策略包。"""

from .base import RecallHit, RecallOutcome, RecallStrategy
from .all import AllRecall
from .bm25 import Bm25Recall
from .embedding import EmbeddingRecall
from .fallback import fallback_hits, is_inventory_query
from .fuzzy import FuzzyRecall
from .hybrid import HybridRRF
from .llm import LlmRecall
from .render import (
    clip_by_budget,
    merge_context,
    render_detail_block,
    render_recent_sources,
    render_statement_block,
    should_prefix_happened_on,
    truncate_source,
)

METHODS = ("fuzzy", "bm25", "llm", "all", "embedding", "hybrid")

__all__ = [
    "RecallHit",
    "RecallOutcome",
    "RecallStrategy",
    "FuzzyRecall",
    "Bm25Recall",
    "EmbeddingRecall",
    "HybridRRF",
    "LlmRecall",
    "AllRecall",
    "is_inventory_query",
    "fallback_hits",
    "render_statement_block",
    "render_detail_block",
    "should_prefix_happened_on",
    "clip_by_budget",
    "merge_context",
    "render_recent_sources",
    "truncate_source",
    "METHODS",
]
