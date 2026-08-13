"""向量召回：FakeEmbedder + 失效补算 + method 门禁。"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from atom_memory.api.deps import get_embedder, get_llm
from atom_memory.db import get_session
from atom_memory.main import app
from atom_memory.models import Atom, AtomEmbedding, AtomKind, AtomStatus, Space
from atom_memory.recall.base import RecallHit
from atom_memory.recall.embedding import EmbeddingRecall, cosine
from atom_memory.recall.hybrid import rrf_fuse
from atom_memory.repositories import embedding_repo
from tests.conftest import FakeLLM


class FakeEmbedder:
    """确定性伪向量：按词袋哈希到固定维，便于测近邻与缓存。"""

    def __init__(self, dim: int = 8, model_tag: str = "fake-emd"):
        self.model_tag = model_tag
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in text.lower().replace("\n", " ").split():
            v[hash(tok) % self.dim] += 1.0
        n = sum(x * x for x in v) ** 0.5
        if n > 0:
            v = [x / n for x in v]
        return v


@pytest.fixture()
def emb_env(fake_llm: FakeLLM):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    embedder = FakeEmbedder()

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_embedder] = lambda: embedder
    with TestClient(app) as client:
        yield client, embedder, engine
    app.dependency_overrides.clear()


def _ops(*items: dict) -> str:
    return json.dumps({"operations": list(items)}, ensure_ascii=False)


def _space_and_atoms(client: TestClient, fake_llm: FakeLLM) -> str:
    uid = client.post("/spaces", json={}).json()["uid"]
    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "user: I love hiking in mountains"},
    )
    src1 = r.json()["id"]
    fake_llm.responses.append(
        _ops(
            {
                "op": "upsert",
                "key": "hobby-hike",
                "kind": "belief",
                "statement": "User enjoys hiking in mountains",
                "detail": "Prefers alpine trails on weekends",
                "change_reason": "from dialogue",
                "source_ids": [src1],
            }
        )
    )
    run = client.post(f"/spaces/{uid}/consolidate", json={}).json()
    assert run["status"] == "succeeded", run

    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "user: I bake sourdough bread"},
    )
    src2 = r.json()["id"]
    fake_llm.responses.append(json.dumps({"read": []}))
    fake_llm.responses.append(
        _ops(
            {
                "op": "upsert",
                "key": "hobby-bake",
                "kind": "belief",
                "statement": "User bakes sourdough bread",
                "detail": "Keeps a starter in the fridge",
                "change_reason": "from dialogue",
                "source_ids": [src2],
            }
        )
    )
    run = client.post(f"/spaces/{uid}/consolidate", json={}).json()
    assert run["status"] == "succeeded", run
    return uid


def test_default_recall_still_bm25(emb_env, fake_llm):
    client, _, _ = emb_env
    uid = _space_and_atoms(client, fake_llm)
    r = client.post(
        f"/spaces/{uid}/recall",
        json={"query": "sourdough baking"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == "bm25"
    assert body["hits"]


def test_embedding_recall_semantic_neighbor(emb_env, fake_llm):
    client, embedder, _ = emb_env
    uid = _space_and_atoms(client, fake_llm)
    r = client.post(
        f"/spaces/{uid}/recall",
        json={
            "query": "hiking mountains",
            "method": "embedding",
            "max_atoms": 1,
        },
    )
    assert r.status_code == 200, r.text
    keys = [h["key"] for h in r.json()["hits"]]
    assert keys[0] == "hobby-hike"
    assert embedder.calls


def test_upsert_invalidates_then_reembed(emb_env, fake_llm):
    client, embedder, engine = emb_env
    uid = _space_and_atoms(client, fake_llm)
    r1 = client.post(
        f"/spaces/{uid}/recall",
        json={"query": "hiking", "method": "embedding", "max_atoms": 2},
    )
    assert r1.status_code == 200
    with Session(engine) as session:
        assert len(session.exec(select(AtomEmbedding)).all()) >= 2
    calls_after_first = len(embedder.calls)

    r = client.post(
        f"/spaces/{uid}/sources",
        json={"kind": "turn", "content": "user: actually alpine scrambling"},
    )
    src3 = r.json()["id"]
    fake_llm.responses.append(json.dumps({"read": ["hobby-hike"]}))
    fake_llm.responses.append(
        _ops(
            {
                "op": "upsert",
                "key": "hobby-hike",
                "kind": "belief",
                "statement": "User loves alpine scrambling",
                "detail": "Uses crampons in winter",
                "change_reason": "update",
                "source_ids": [src3],
            }
        )
    )
    run = client.post(f"/spaces/{uid}/consolidate", json={}).json()
    assert run["status"] == "succeeded", run

    with Session(engine) as session:
        atom_ids = {
            a.key: a.id
            for a in session.exec(select(Atom)).all()
        }
        hike_id = atom_ids["hobby-hike"]
        cached = {
            row.atom_id
            for row in session.exec(select(AtomEmbedding)).all()
        }
        assert hike_id not in cached

    r2 = client.post(
        f"/spaces/{uid}/recall",
        json={
            "query": "alpine scrambling",
            "method": "embedding",
            "max_atoms": 1,
        },
    )
    assert r2.status_code == 200
    assert len(embedder.calls) > calls_after_first
    assert r2.json()["hits"][0]["key"] == "hobby-hike"


def test_embedding_without_embedder_returns_422(fake_llm):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_embedder] = lambda: None
    try:
        with TestClient(app) as client:
            uid = client.post("/spaces", json={}).json()["uid"]
            r = client.post(
                f"/spaces/{uid}/recall",
                json={"query": "x", "method": "embedding"},
            )
            assert r.status_code == 422
            r2 = client.post(
                f"/spaces/{uid}/recall",
                json={"query": "x", "method": "hybrid"},
            )
            assert r2.status_code == 422
            r3 = client.post(
                f"/spaces/{uid}/recall",
                json={"query": "x", "method": "bm25"},
            )
            assert r3.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_hybrid_returns_hits(emb_env, fake_llm):
    client, _, _ = emb_env
    uid = _space_and_atoms(client, fake_llm)
    r = client.post(
        f"/spaces/{uid}/recall",
        json={
            "query": "sourdough bread baking",
            "method": "hybrid",
            "max_atoms": 2,
        },
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["hits"]) >= 1


def test_cosine_and_rrf_unit():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    now = datetime.now(timezone.utc)

    def atom(key: str) -> Atom:
        return Atom(
            id=abs(hash(key)) % 10000,
            space_id=1,
            kind=AtomKind.belief,
            key=key,
            statement=key,
            detail="",
            status=AtomStatus.active,
            schema_version=1,
            created_at=now,
            updated_at=now,
        )

    a, b, c = atom("a"), atom("b"), atom("c")
    fused = rrf_fuse(
        [
            [RecallHit(a, 1.0), RecallHit(b, 0.5)],
            [RecallHit(b, 0.9), RecallHit(c, 0.8)],
        ],
        limit=3,
    )
    assert [h.atom.key for h in fused][0] == "b"


def test_embedding_recall_direct_lazy():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    embedder = FakeEmbedder()
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        space = Space(uid="testspace1")
        session.add(space)
        session.commit()
        session.refresh(space)
        atom = Atom(
            space_id=space.id,
            kind=AtomKind.belief,
            key="k1",
            statement="cats prefer sunny windows",
            detail="afternoon nap",
            status=AtomStatus.active,
            schema_version=1,
            created_at=now,
            updated_at=now,
        )
        session.add(atom)
        session.commit()
        session.refresh(atom)

        strat = EmbeddingRecall(session, embedder, embed_detail_chars=200)
        out1 = strat.retrieve([atom], "sunny window cats", 5)
        assert out1.hits
        assert len(embedder.calls) == 2

        calls_before = len(embedder.calls)
        strat.retrieve([atom], "sunny window cats", 5)
        assert len(embedder.calls) == calls_before + 1
        assert len(embedder.calls[-1]) == 1

        embedding_repo.invalidate_for_atom(session, atom.id)
        session.commit()
        calls_before = len(embedder.calls)
        strat.retrieve([atom], "sunny", 5)
        assert len(embedder.calls) == calls_before + 2
