"""Offline API latency baseline. Uses a disposable file-backed SQLite database.

Run from the repository: python scripts/benchmark_local.py --out report.json
TestClient includes routing/serialization, but no socket, TLS, or model latency.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[100, 1000, 10000])
    parser.add_argument('--requests', type=int, default=60)
    args = parser.parse_args()
    if args.requests < 20 or min(args.sizes) < 1:
        parser.error('use at least 20 requests and positive sizes')
    with tempfile.TemporaryDirectory(prefix='atommem-perf-') as temp:
        os.environ['ATOMMEM_DATABASE_URL'] = 'sqlite:///' + str(Path(temp) / 'benchmark.db').replace('\\', '/')
        os.environ['ATOMMEM_API_KEY'] = ''
        from fastapi.testclient import TestClient
        from sqlmodel import Session
        from atom_memory.api.deps import get_llm, get_embedder
        from atom_memory.db import engine
        from atom_memory.main import app
        from atom_memory.models import Atom, AtomKind, Space

        class NoModel:
            def complete(self, *a, **kw):
                raise AssertionError('benchmark must not invoke a model')

        app.dependency_overrides[get_llm] = lambda: NoModel()
        app.dependency_overrides[get_embedder] = lambda: None
        rows = []
        raw = []

        def measure(label, operation, validate, warmups=5):
            for i in range(warmups):
                validate(operation(i))
            times = []
            errors = []
            started = time.perf_counter()
            for i in range(args.requests):
                tick = time.perf_counter()
                try:
                    response = operation(i + warmups)
                    elapsed = (time.perf_counter() - tick) * 1000
                    validate(response)
                except Exception as exc:
                    elapsed = (time.perf_counter() - tick) * 1000
                    errors.append({'request': i, 'error': str(exc)})
                times.append(elapsed)
            wall = time.perf_counter() - started
            ordered = sorted(times)
            row = dict(label=label, requests=len(times), errors=errors,
                       mean_ms=statistics.mean(times), p50_ms=statistics.median(times),
                       p95_ms=ordered[math.ceil(.95 * len(times)) - 1],
                       max_ms=max(times), sequential_requests_per_second=len(times)/wall)
            rows.append(row)
            raw.append(dict(label=label, milliseconds=times))
            print(json.dumps(row), flush=True)

        def ok(response):
            assert response.status_code == 200, response.text

        try:
            with TestClient(app) as client:
                for size in args.sizes:
                    uid = f'benchmark-{size}'
                    with Session(engine) as session:
                        space = Space(uid=uid)
                        session.add(space)
                        session.flush()
                        for i in range(size):
                            session.add(Atom(space_id=space.id, key=f'fact-{i}', kind=AtomKind.belief,
                                statement=f'Person {i} prefers topic{i % 100} coffee in city{i % 50}.',
                                detail=f'Person {i} studies topic{i % 100} on weekends.'))
                        session.commit()
                    for method in ('bm25', 'fuzzy'):
                        for budget in (400, 1600):
                            def validate(response, budget=budget):
                                ok(response)
                                result = response.json()
                                assert 0 < result['chars_used'] <= budget, result
                                assert 0 < len(result['hits']) <= 5, result
                            measure(f'recall/{size}/{method}/{budget}',
                                lambda i, method=method, budget=budget: client.post(f'/spaces/{uid}/recall', json={
                                    'query': f'topic{i % 100} coffee', 'method': method,
                                    'budget_chars': budget, 'max_atoms': 5, 'include_recent_sources': False}), validate)
                    # Independent spaces remain in the database: per-space cardinality is explicit.
                uid = 'benchmark-writes'
                ok(client.post('/spaces', json={'uid': uid}))
                measure('source/new', lambda i: client.post(f'/spaces/{uid}/sources', json={
                    'kind': 'turn', 'content': 'Synthetic performance source.', 'idempotency_key': f'write-{i}'}), ok)
                response = client.post(f'/spaces/{uid}/sources', json={
                    'kind': 'turn', 'content': 'Idempotent synthetic source.', 'idempotency_key': 'retry'})
                ok(response)
                source_id = response.json()['id']
                def same_source(response):
                    ok(response)
                    assert response.json()['id'] == source_id
                measure('source/retry', lambda i: client.post(f'/spaces/{uid}/sources', json={
                    'kind': 'turn', 'content': 'Idempotent synthetic source.', 'idempotency_key': 'retry'}), same_source)
                measure('source/status', lambda i: client.get(f'/spaces/{uid}/sources/{source_id}'), ok)
        finally:
            app.dependency_overrides.clear()
            engine.dispose()

    report = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(), head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=sys.version, platform=platform.platform(), cpu=platform.processor(), logical_cpus=os.cpu_count(),
        packages={name: importlib.metadata.version(name) for name in ('fastapi', 'sqlmodel', 'sqlalchemy', 'httpx', 'snowballstemmer')},
        protocol=dict(sizes=args.sizes, requests=args.requests, warmups=5, concurrency=1,
            database='disposable file-backed SQLite, default journal; all sizes coexist in separate spaces',
            fixture='deterministic synthetic English L1 facts; no evidence, revisions, links, or pending sources',
            query='topic{i % 100} coffee; max_atoms=5; include_recent_sources=false',
            timing='TestClient API response time; validation excluded from latency, included in throughput; nearest-rank p95',
            limitations='No network, model, PostgreSQL, concurrent load, production data distribution, or SLO certification'),
        results=rows, raw=raw)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return int(any(row['errors'] for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
