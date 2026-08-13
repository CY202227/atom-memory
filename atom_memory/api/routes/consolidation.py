"""固化路由：触发"睡眠"与查看运行日志；综合层独立触发。

上游想什么时候固化都行（日记生成后 / 会话结束 / cron），
本服务不内置调度。综合（synthesize）宜用定时任务，勿挂进每次 consolidate。
"""

from fastapi import APIRouter, Depends
from sqlmodel import Session

from ...config import settings
from ...consolidation.engine import ConsolidationEngine
from ...consolidation.synthesis import SynthesisEngine
from ...db import get_session
from ...llm import ChatLLM
from ...locks import space_write_lock
from ...models import ConsolidationRun, Space
from ...repositories import run_repo
from .. import schemas
from ..deps import get_engine, get_llm, get_space, require_api_key

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/spaces/{space_uid}/consolidate", response_model=ConsolidationRun)
def consolidate(
    payload: schemas.ConsolidateRequest,
    space: Space = Depends(get_space),
    session: Session = Depends(get_session),
    engine: ConsolidationEngine = Depends(get_engine),
):
    # 与 delete-by-ref 共用 space 写锁：固化引用的 source 不能被并发删除
    with space_write_lock(space.id):
        return engine.run(
            session,
            space,
            trigger=payload.trigger,
            max_sources=payload.max_sources or settings.consolidate_max_sources,
        )


@router.post("/spaces/{space_uid}/synthesize", response_model=ConsolidationRun)
def synthesize(
    payload: schemas.SynthesizeRequest,
    space: Space = Depends(get_space),
    session: Session = Depends(get_session),
    llm: ChatLLM = Depends(get_llm),
):
    """从已有 atom 归纳更高层认识；独立于 consolidate，宜定时触发。"""
    engine = SynthesisEngine(llm)
    with space_write_lock(space.id):
        return engine.run(
            session,
            space,
            trigger=payload.trigger,
            max_read=payload.max_read,
        )


@router.get(
    "/spaces/{space_uid}/runs",
    response_model=list[ConsolidationRun],
    tags=["admin"],
)
def list_runs(space: Space = Depends(get_space), session: Session = Depends(get_session)):
    return run_repo.list_by_space(session, space.id)
