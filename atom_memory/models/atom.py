"""atom 表：一条可注入的认识断言（Atom-first）。

statement（≤80 字）是默认召回注入单元；detail（≤300 字）供按需展开。
六型 kind 分类。(space_id, key) 唯一。只能经固化引擎或回滚改写。
"""

from datetime import date, datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, Text, UniqueConstraint
from sqlmodel import Field, SQLModel

from .base import SCHEMA_VERSION, utcnow


class AtomKind(str, Enum):
    lesson = "lesson"
    event = "event"
    person = "person"
    belief = "belief"
    procedure = "procedure"
    self = "self"


class AtomStatus(str, Enum):
    active = "active"
    archived = "archived"


class Atom(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("space_id", "key"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    space_id: int = Field(foreign_key="space.id", index=True)
    kind: AtomKind = Field(index=True)
    key: str = Field(index=True)
    statement: str  # ≤80；默认注入
    detail: str = Field(default="", sa_column=Column(Text, nullable=False))
    happened_on: Optional[date] = None
    confidence: Optional[float] = Field(default=None)
    # L1=事实原子 L2=场景综合 L3=稳定画像；L0 是 source，不在本表
    memory_layer: int = Field(default=1, index=True)
    status: AtomStatus = Field(default=AtomStatus.active, index=True)
    schema_version: int = Field(default=SCHEMA_VERSION)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
