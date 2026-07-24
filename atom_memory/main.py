"""应用入口：装配 FastAPI、建表、挂路由。"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
import uvicorn

from .api.router import api_router
from .db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="atom-memory",
    description=(
        "Atom-first 长期记忆服务：不可变 source 经 LLM 固化为短 statement 原子；"
        "召回默认注入 statement（字符预算）；按需展开 detail/证据；"
        "/ui 预览；/chat 失忆调试对话。"
    ),
    lifespan=lifespan,
)
app.include_router(api_router)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    # reload 模式要求传应用导入串而非实例，否则改代码不会自动重启
    uvicorn.run("atom_memory.main:app", host="0.0.0.0", port=8020, reload=True)
