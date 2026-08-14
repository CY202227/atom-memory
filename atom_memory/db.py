from sqlalchemy import event, text
from sqlmodel import Session, SQLModel, create_engine

from .config import settings

_is_sqlite = settings.database_url.startswith("sqlite")
_connect_args = {"check_same_thread": False} if _is_sqlite else {}
engine = create_engine(settings.database_url, connect_args=_connect_args)

if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _set_sqlite_foreign_keys(dbapi_connection, connection_record):
        del connection_record  # Unused.
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _ensure_atom_memory_layer_column() -> None:
    """存量 SQLite：create_all 不 ALTER，补 memory_layer 列。"""
    if not _is_sqlite:
        return
    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(atom)")).fetchall()
        if not rows:
            return
        names = {r[1] for r in rows}
        if "memory_layer" not in names:
            conn.execute(
                text(
                    "ALTER TABLE atom ADD COLUMN memory_layer "
                    "INTEGER NOT NULL DEFAULT 1"
                )
            )


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    _ensure_atom_memory_layer_column()


def get_session():
    with Session(engine) as session:
        yield session
