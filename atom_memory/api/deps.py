"""API 依赖装配：鉴权、DB 会话、space 解析、LLM 与固化引擎、召回策略。

Fake/Real 分叉只发生在这里（测试经 dependency_overrides 注入 FakeLLM），
路由代码不感知实现选择。
"""

from fastapi import Depends, Header, HTTPException
from sqlmodel import Session

from ..config import settings
from ..consolidation.engine import ConsolidationEngine
from ..db import get_session
from ..embedding import OpenAICompatEmbedder
from ..llm import ChatLLM, OpenAICompatLLM
from ..models import Space
from ..recall import Bm25Recall, FuzzyRecall, LlmRecall, RecallStrategy
from ..repositories import space_repo


def require_api_key(x_api_key: str = Header(default="")) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid API key")


def get_llm() -> ChatLLM:
    return OpenAICompatLLM(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        timeout=settings.llm_timeout_seconds,
        trust_env=settings.llm_trust_env,
        extra_body=settings.llm_extra_body,
    )


def get_embedder() -> OpenAICompatEmbedder | None:
    base = (settings.embedder_api_base or "").strip()
    model = (settings.embedder_model or "").strip()
    if not base or not model:
        return None
    return OpenAICompatEmbedder(
        api_base=base,
        api_key=settings.embedder_api_key or "EMPTY",
        model=model,
        timeout=settings.embedder_timeout_seconds,
        trust_env=settings.embedder_trust_env,
    )


def get_engine(llm: ChatLLM = Depends(get_llm)) -> ConsolidationEngine:
    return ConsolidationEngine(llm)


def get_space(space_uid: str, session: Session = Depends(get_session)) -> Space:
    space = space_repo.get_by_uid(session, space_uid)
    if space is None:
        raise HTTPException(status_code=404, detail="space not found")
    return space


def build_recall_strategy(
    method: str,
    llm: ChatLLM,
    *,
    session: Session | None = None,
    embedder: OpenAICompatEmbedder | None = None,
) -> RecallStrategy:
    if method == "fuzzy":
        return FuzzyRecall()
    if method == "bm25":
        return Bm25Recall()
    if method == "llm":
        return LlmRecall(llm)
    if method == "all":
        from ..recall.all import AllRecall

        return AllRecall()
    if method in ("embedding", "hybrid"):
        if embedder is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "embedding/hybrid requires ATOMMEM_EMBEDDER_API_BASE "
                    "and ATOMMEM_EMBEDDER_MODEL"
                ),
            )
        if session is None:
            raise HTTPException(
                status_code=500,
                detail="embedding/hybrid requires a DB session",
            )
        from ..recall.embedding import EmbeddingRecall
        from ..recall.hybrid import HybridRRF

        if method == "embedding":
            return EmbeddingRecall(
                session,
                embedder,
                embed_detail_chars=settings.embed_detail_chars,
            )
        return HybridRRF(
            session,
            embedder,
            embed_detail_chars=settings.embed_detail_chars,
        )
    raise HTTPException(status_code=422, detail=f"unknown recall method: {method}")
