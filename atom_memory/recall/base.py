"""召回策略契约。"""

from dataclasses import dataclass, field
from typing import Optional, Protocol
from datetime import date

from ..models import Atom


@dataclass
class RecallHit:
    atom: Atom
    score: Optional[float] = None
    source_anchor: Optional[date] = None


@dataclass
class RecallOutcome:
    hits: list[RecallHit] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class RecallStrategy(Protocol):
    def retrieve(self, atoms: list[Atom], query: str, max_atoms: int) -> RecallOutcome: ...
