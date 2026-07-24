"""evidence 表：出处链（原子修订 ↔ source 多对多）。"""

from typing import Optional

from sqlmodel import Field, SQLModel


class Evidence(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    revision_id: int = Field(foreign_key="atomrevision.id", index=True)
    source_id: int = Field(foreign_key="source.id", index=True)
    note: Optional[str] = Field(default=None)
