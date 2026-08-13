"""atom_embedding 表：语义召回通道的 atom 级向量（纯派生物）。

检索面：statement + detail 截断。JSON 存向量、内存余弦。
固化/删除只做失效（删行）；召回时惰性补算。
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel, UniqueConstraint

from .base import utcnow


class AtomEmbedding(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("atom_id"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    atom_id: int = Field(foreign_key="atom.id", index=True)
    space_id: int = Field(foreign_key="space.id", index=True)
    model_tag: str
    dim: int
    vector: list[float] = Field(sa_column=Column(JSON, nullable=False))
    updated_at: datetime = Field(default_factory=utcnow)
