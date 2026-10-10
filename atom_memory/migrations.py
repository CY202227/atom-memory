"""Ordered, additive schema upgrades for SQLite and PostgreSQL."""
from sqlalchemy import inspect, text

LATEST_VERSION = 2


def _add_columns(conn, table: str, columns: dict[str, str]) -> None:
    inspector = inspect(conn)
    if not inspector.has_table(table):
        return
    existing = {column["name"] for column in inspector.get_columns(table)}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {definition}"))


def upgrade_atom(conn) -> None:
    _add_columns(conn, "atom", {
        "happened_on": "DATE",
        "memory_layer": "INTEGER NOT NULL DEFAULT 1",
    })
    _add_columns(conn, "atomrevision", {"happened_on": "DATE"})
    if inspect(conn).has_table("atom"):
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_atom_memory_layer ON atom (memory_layer)"))


def upgrade_source(conn) -> None:
    _add_columns(conn, "source", {"idempotency_key": "VARCHAR(128)"})
    if inspect(conn).has_table("source"):
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_source_retry_key ON source (space_id, idempotency_key)"))


MIGRATIONS = ((1, upgrade_atom), (2, upgrade_source))


def migrate(engine) -> None:
    with engine.begin() as conn:
        if conn.dialect.name == "postgresql":
            # Serialize startup migrations across service instances.
            conn.execute(text("SELECT pg_advisory_xact_lock(17012026)"))
        elif conn.dialect.name == "sqlite":
            # Lock before reading the version so concurrent startups cannot race.
            conn.exec_driver_sql("BEGIN IMMEDIATE")
        else:
            raise RuntimeError("schema upgrades support SQLite and PostgreSQL only")
        conn.execute(text("CREATE TABLE IF NOT EXISTS atom_memory_schema (version INTEGER PRIMARY KEY)"))
        current = conn.execute(text("SELECT MAX(version) FROM atom_memory_schema")).scalar() or 0
        if current > LATEST_VERSION:
            raise RuntimeError(f"database schema {current} is newer than supported {LATEST_VERSION}")
        for version, upgrade in MIGRATIONS:
            if version > current:
                upgrade(conn)
                conn.execute(text("INSERT INTO atom_memory_schema (version) VALUES (:version)"), {"version": version})
