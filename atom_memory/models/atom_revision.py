"""atom_revision 表：原子认识的修订历史（可解释、可回滚、可审计）。"""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, Text, UniqueConstraint
from sqlmodel import Field, SQLModel

from .base import utcnow


class RevisionTrigger(str, Enum):
    consolidation = "consolidation"
    manual = "manual"
    rollback = "rollback"


class AtomRevision(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("atom_id", "seq"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    atom_id: int = Field(foreign_key="atom.id", index=True)
    seq: int
    statement: str
    detail: str = Field(default="", sa_column=Column(Text, nullable=False))
    happened_on: Optional[date] = None
    change_reason: str
    trigger: RevisionTrigger
    run_id: Optional[int] = Field(default=None, foreign_key="consolidationrun.id")
    created_at: datetime = Field(default_factory=utcnow)
