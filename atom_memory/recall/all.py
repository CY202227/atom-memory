"""全量召回：不按 query 截断，按更新时间倒序返回（评测摸天花板用）。"""

from ..models import Atom
from .base import RecallHit, RecallOutcome


class AllRecall:
    """返回至多 max_atoms 条；忽略 query 相关性（用于 retrieval 上限）。"""

    def retrieve(self, atoms: list[Atom], query: str, max_atoms: int) -> RecallOutcome:
        del query  # 全量模式不打分
        ordered = sorted(
            atoms,
            key=lambda a: a.updated_at or a.created_at,
            reverse=True,
        )
        hits = [RecallHit(atom=a, score=1.0) for a in ordered[:max_atoms]]
        return RecallOutcome(hits=hits)
