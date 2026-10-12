from datetime import datetime
from sqlalchemy import DateTime
from sqlmodel import SQLModel, Session, create_engine
from atom_memory.models import Space, Source, SourceKind


def test_persistent_timestamps_use_explicit_naive_storage():
    # SQLModel newer releases change bare datetime inference; the storage contract
    # must stay UTC naive for existing SQLite and PostgreSQL databases.
    for table in SQLModel.metadata.tables.values():
        for column in table.columns:
            if isinstance(column.type, DateTime):
                assert column.type.timezone is False, f"{table.name}.{column.name}"
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        space = Space(uid="timestamp-contract")
        session.add(space)
        session.commit()
        source = Source(space_id=space.id, kind=SourceKind.turn,
                        content="fact", content_hash="test", occurred_at=datetime(2026, 10, 10, 9))
        session.add(source)
        session.commit()
        session.refresh(source)
        assert source.occurred_at == datetime(2026, 10, 10, 9)
        assert source.ingested_at.tzinfo is None
    engine.dispose()
