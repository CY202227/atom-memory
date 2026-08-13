"""API 出入参 DTO：对外契约的形状定义（详见 docs/contract.md）。"""

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ..models import AtomKind, AtomStatus, SourceKind


class SpaceCreate(BaseModel):
    uid: Optional[str] = Field(default=None, min_length=8, max_length=64)
    owner_id: Optional[str] = None
    subject_id: Optional[str] = None


class SourceIngest(BaseModel):
    kind: SourceKind
    content: str = Field(min_length=1)
    occurred_at: Optional[datetime] = None
    external_ref: Optional[dict] = None
    salience: float = Field(default=0.0, ge=0.0, le=1.0)


class DeleteBySourceRequest(BaseModel):
    external_ref: dict

    @field_validator("external_ref")
    @classmethod
    def _non_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError("external_ref 至少需要一个键（推荐 system + session_id）")
        return v


class DeleteBySourcePreview(BaseModel):
    matched_sources: int
    matched_source_ids: list[int]
    atoms_to_delete: list[str]
    atoms_to_reconsolidate: list[str]
    atoms_derived_affected: list[str] = []


class DeleteBySourceResult(BaseModel):
    deleted_sources: int
    deleted_atoms: list[str]
    reconsolidated_atoms: list[str]
    derived_affected_atoms: list[str] = []
    run_id: Optional[int] = None


class AtomDetail(BaseModel):
    id: int
    space_id: int
    kind: AtomKind
    key: str
    statement: str
    detail: str
    happened_on: Optional[date] = None
    confidence: Optional[float] = None
    status: AtomStatus
    schema_version: int
    created_at: datetime
    updated_at: datetime
    evidence_dates: list[str]
    revisions: Optional[list[dict]] = None
    evidence: Optional[list[dict]] = None


class AtomListResponse(BaseModel):
    count: int
    page: int
    page_size: int
    results: list[dict]


class RollbackRequest(BaseModel):
    seq: int


class ConsolidateRequest(BaseModel):
    trigger: str = "manual"
    max_sources: Optional[int] = Field(default=None, ge=1, le=200)


class SynthesizeRequest(BaseModel):
    trigger: str = "synthesize"
    max_read: int = Field(default=8, ge=2, le=20)


class RecallRequest(BaseModel):
    query: str = Field(min_length=1)
    method: Literal["fuzzy", "bm25", "llm", "all", "embedding", "hybrid"] = "bm25"
    max_atoms: int = Field(default=5, ge=1, le=2000)
    budget_chars: int = Field(default=400, ge=40, le=200000)
    include_recent_sources: bool = True
    # statement=仅断言（默认）；full=含 detail，仍受 budget
    detail: Literal["statement", "full"] = "statement"
    # 0=不扩展（默认）；1=沿 about/derived_from 补一跳邻居
    neighbor_hops: int = Field(default=0, ge=0, le=1)


class AtomNeighborsResponse(BaseModel):
    key: str
    out: list[dict]
    incoming: list[dict]


class RecallHitOut(BaseModel):
    key: str
    kind: AtomKind
    statement: str
    score: Optional[float] = None
    happened_on: Optional[date] = None
    detail: Optional[str] = None  # 仅 detail=full 时填充


class RecallResponse(BaseModel):
    method: str
    hits: list[RecallHitOut]
    context_block: str
    chars_used: int = 0
    atoms_clipped: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ExpandRequest(BaseModel):
    keys: list[str] = Field(min_length=1, max_length=10)
    with_evidence: bool = False
    budget_chars: int = Field(default=1200, ge=100, le=8000)


class ExpandHitOut(BaseModel):
    key: str
    kind: AtomKind
    statement: str
    detail: str
    happened_on: Optional[date] = None
    evidence_excerpts: list[dict] = []


class ExpandResponse(BaseModel):
    hits: list[ExpandHitOut]
    missing: list[str]
    context_block: str
