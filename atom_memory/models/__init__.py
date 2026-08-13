"""数据模型包：一张表一个文件，此处统一导出。

依赖关系：
space → source → consolidationrun → atom → atomrevision ← evidence ← source
atom ↔ atom_link（类型化边）
"""

from .atom import Atom, AtomKind, AtomStatus
from .atom_embedding import AtomEmbedding
from .atom_link import AtomLink, AtomLinkKind
from .atom_revision import AtomRevision, RevisionTrigger
from .base import SCHEMA_VERSION, utcnow
from .consolidation_run import ConsolidationRun, RunStatus
from .evidence import Evidence
from .source import Source, SourceKind, SourceStatus
from .space import Space, new_uid

__all__ = [
    "SCHEMA_VERSION",
    "utcnow",
    "Space",
    "new_uid",
    "Source",
    "SourceKind",
    "SourceStatus",
    "Atom",
    "AtomKind",
    "AtomStatus",
    "AtomEmbedding",
    "AtomLink",
    "AtomLinkKind",
    "AtomRevision",
    "RevisionTrigger",
    "Evidence",
    "ConsolidationRun",
    "RunStatus",
]
