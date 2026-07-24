"""consolidationrun 表："睡眠"固化的运行日志。"""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import JSON, Column, Text
from sqlmodel import Field, SQLModel

from .base import utcnow


class RunStatus(str, Enum):
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class ConsolidationRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    space_id: int = Field(foreign_key="space.id", index=True)
    trigger: str = Field(default="manual")
    status: RunStatus = Field(default=RunStatus.running)
    source_ids: Optional[list] = Field(default=None, sa_column=Column(JSON))
    atoms_touched: Optional[list] = Field(default=None, sa_column=Column(JSON))
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    error: Optional[str] = Field(default=None, sa_column=Column(Text))
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: Optional[datetime] = Field(default=None)
