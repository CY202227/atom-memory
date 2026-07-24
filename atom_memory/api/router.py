"""路由汇聚。"""

from fastapi import APIRouter

from .routes import atoms, chat_demo, consolidation, recall, sources, spaces, ui

api_router = APIRouter()
api_router.include_router(spaces.router, tags=["spaces"])
api_router.include_router(sources.router, tags=["sources"])
api_router.include_router(atoms.router, tags=["atoms"])
api_router.include_router(consolidation.router, tags=["consolidation"])
api_router.include_router(recall.router, tags=["recall"])
api_router.include_router(ui.router, tags=["ui"])
api_router.include_router(chat_demo.router)
