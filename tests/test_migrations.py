import pytest
from sqlalchemy import inspect, text
from sqlmodel import create_engine
from atom_memory.migrations import migrate, LATEST_VERSION


def test_upgrade_preserves_old_rows_and_is_repeatable(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE atom (id INTEGER PRIMARY KEY, statement TEXT)"))
        conn.execute(text("INSERT INTO atom VALUES (1, 'old fact')"))
        conn.execute(text("CREATE TABLE source (id INTEGER PRIMARY KEY, space_id INTEGER)"))
        conn.execute(text("INSERT INTO source VALUES (1, 7)"))
        conn.execute(text("CREATE TABLE atomrevision (id INTEGER PRIMARY KEY, statement TEXT)"))
        conn.execute(text("INSERT INTO atomrevision VALUES (1, 'old revision')"))
    migrate(engine)
    migrate(engine)
    with engine.connect() as conn:
        assert conn.execute(text("SELECT statement, memory_layer FROM atom")).one() == ("old fact", 1)
        assert conn.execute(text("SELECT id, idempotency_key FROM source")).one() == (1, None)
        assert conn.execute(text("SELECT statement, happened_on FROM atomrevision")).one() == ("old revision", None)
        assert conn.execute(text("SELECT MAX(version) FROM atom_memory_schema")).scalar() == LATEST_VERSION
        assert {"memory_layer", "happened_on"} <= {c["name"] for c in inspect(conn).get_columns("atom")}
    engine.dispose()


def test_future_schema_is_rejected(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'future.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE atom_memory_schema (version INTEGER PRIMARY KEY)"))
        conn.execute(text("INSERT INTO atom_memory_schema VALUES (999)"))
    with pytest.raises(RuntimeError, match="newer"):
        migrate(engine)
    engine.dispose()


def test_failed_migration_does_not_advance_version(tmp_path, monkeypatch):
    from atom_memory import migrations
    engine = create_engine(f"sqlite:///{tmp_path / 'failure.db'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE source (id INTEGER PRIMARY KEY, space_id INTEGER)"))
    def fail(conn):
        conn.execute(text("ALTER TABLE source ADD COLUMN should_rollback TEXT"))
        raise RuntimeError("migration failure")
    monkeypatch.setattr(migrations, "MIGRATIONS", ((1, fail),))
    with pytest.raises(RuntimeError, match="failure"):
        migrate(engine)
    with engine.connect() as conn:
        assert "should_rollback" not in {c["name"] for c in inspect(conn).get_columns("source")}
    engine.dispose()


def test_interrupted_run_expires_and_keeps_pending_source():
    from datetime import timedelta
    from sqlalchemy.pool import StaticPool
    from sqlmodel import SQLModel, Session
    from atom_memory.models import Space, Source, SourceKind, SourceStatus, RunStatus, utcnow
    from atom_memory.repositories import run_repo
    engine = create_engine("sqlite://", poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        space = Space(uid="interrupted-space")
        session.add(space)
        session.commit()
        source = Source(space_id=space.id, kind=SourceKind.correction, content="correction", content_hash="x")
        session.add(source)
        session.commit()
        run = run_repo.create(session, space.id, "correction", [source.id])
        run.started_at = utcnow() - timedelta(hours=1)
        session.add(run)
        session.commit()
        assert run_repo.find_active(session, space.id) is None
        session.refresh(run)
        session.refresh(source)
        assert run.status == RunStatus.failed
        assert source.status == SourceStatus.pending
    engine.dispose()
