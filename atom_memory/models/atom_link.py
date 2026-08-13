"""atom_link 表：类型化有向边（出处 / 导航 / 冲突）。

三种边契约：
- derived_from：A 由 B 推出；承载失效传播
- about：弱关联，仅导航扩展
- contradicts：显式冲突，可查询
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel, UniqueConstraint

from .base import utcnow


class AtomLinkKind(str, Enum):
    derived_from = "derived_from"
    about = "about"
    contradicts = "contradicts"


class AtomLink(SQLModel, table=True):
    __table_args__ = (
        UniqueConstraint("src_atom_id", "dst_atom_id", "kind"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    space_id: int = Field(foreign_key="space.id", index=True)
    src_atom_id: int = Field(foreign_key="atom.id", index=True)
    dst_atom_id: int = Field(foreign_key="atom.id", index=True)
    kind: AtomLinkKind = Field(index=True)
    note: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow)
