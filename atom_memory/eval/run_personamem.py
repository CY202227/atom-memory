"""独立入口：跑 PersonaMem 并打印得分。

用法::

    # 迷你集（离线，oracle 固化 + 检索命中分）
    python -m atom_memory.eval.run_personamem --mini

    # 选择题（需 .env LLM）
    python -m atom_memory.eval.run_personamem --mini --score mcq

    # 自由作答 F1
    python -m atom_memory.eval.run_personamem --mini --score qa

    # 限制规模
    python -m atom_memory.eval.run_personamem --score mcq \\
        --max-samples 1 --max-questions 10
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from atom_memory.api.deps import get_llm
from atom_memory.config import settings
from atom_memory.db import get_session
from atom_memory.eval.personamem import (
    aggregate_reports,
    consolidate_sessions_oracle,
    evaluate_personamem_sample,
    format_personamem_scoreboard,
    ingest_personamem_sample,
    load_personamem,
)
from atom_memory.llm.base import ChatResult
from atom_memory.llm.openai_compat import OpenAICompatLLM
from atom_memory.main import app

_FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
_MINI = _FIXTURES / "personamem_mini.json"


class _QueueLLM:
    """oracle 固化用的假 LLM。"""

    def __init__(self) -> None:
        self.responses: list[str] = []

    def complete(
        self, system: str, user: str, response_format: dict | None = None
    ) -> ChatResult:
        del system, user, response_format
        if not self.responses:
            raise RuntimeError("QueueLLM: empty responses")
        return ChatResult(
            text=self.responses.pop(0), prompt_tokens=1, completion_tokens=1
        )


def _build_client(llm) -> TestClient:
    """独立临时 SQLite，不污染业务库。"""
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
    app.dependency_overrides[get_llm] = lambda: llm
    return TestClient(app)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m atom_memory.eval.run_personamem",
        description="Run PersonaMem eval and print a scoreboard.",
    )
    p.add_argument(
        "data",
        nargs="?",
        default="",
        help=(
            "Path to PersonaMem JSON; default requires --mini "
            "or ATOMMEM_PERSONAMEM_PATH"
        ),
    )
    p.add_argument(
        "--mini",
        action="store_true",
        help=f"Use tiny fixture ({_MINI.name})",
    )
    p.add_argument(
        "--score",
        choices=("retrieval", "mcq", "qa"),
        default="retrieval",
        help=(
            "retrieval=金标是否在召回文本; "
            "mcq=LLM 选 A/B/C/D; "
            "qa=LLM 作答后算 F1"
        ),
    )
    p.add_argument(
        "--retrieval-coverage",
        type=float,
        default=1.0,
        help=(
            "仅 --score retrieval：金标 token 覆盖率阈值 "
            "(1.0=硬逻辑；0.6=诊断软计分)"
        ),
    )
    p.add_argument(
        "--consolidate",
        choices=("oracle", "llm"),
        default="oracle",
        help="oracle=按用户句写 atom（快）; llm=真实固化（慢、耗 token）",
    )
    p.add_argument(
        "--method",
        default="bm25",
        choices=("bm25", "fuzzy", "llm", "all", "embedding", "hybrid"),
    )
    p.add_argument("--budget-chars", type=int, default=400)
    p.add_argument("--max-atoms", type=int, default=5)
    p.add_argument(
        "--detail",
        default="statement",
        choices=("statement", "full"),
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-sessions", type=int, default=None)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument(
        "--out",
        default="",
        help="Write JSON report to this path (default: temp file printed)",
    )
    p.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit 1 if accuracy is below this threshold",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mini:
        data_path = _MINI
    elif args.data:
        data_path = Path(args.data)
    else:
        import os

        env_path = os.environ.get("ATOMMEM_PERSONAMEM_PATH", "").strip()
        if env_path:
            data_path = Path(env_path)
        else:
            print(
                "error: provide data path, --mini, or "
                "ATOMMEM_PERSONAMEM_PATH",
                file=sys.stderr,
            )
            return 2
    if not data_path.is_file():
        print(f"error: data file not found: {data_path}", file=sys.stderr)
        return 2

    need_real_llm = args.score in ("mcq", "qa") or args.consolidate == "llm"
    answer_llm = None
    if need_real_llm:
        if not settings.llm_base_url or not settings.llm_model:
            print(
                "error: --score mcq/qa / --consolidate llm 需要 .env 里 "
                "ATOMMEM_LLM_BASE_URL 与 ATOMMEM_LLM_MODEL",
                file=sys.stderr,
            )
            return 2
        answer_llm = OpenAICompatLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key or "EMPTY",
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
            trust_env=settings.llm_trust_env,
            extra_body=settings.llm_extra_body,
        )

    queue_llm = _QueueLLM()
    wire_llm = answer_llm if args.consolidate == "llm" else queue_llm
    client = _build_client(wire_llm)
    try:
        samples = load_personamem(data_path)
        if args.max_samples is not None:
            samples = samples[: args.max_samples]

        mode = f"consolidate={args.consolidate};score={args.score}"
        if args.score == "retrieval":
            mode += f";coverage={args.retrieval_coverage:g}"
        print(
            f"PersonaMem: {data_path} | samples={len(samples)} | {mode} | "
            f"budget={args.budget_chars} detail={args.detail}",
            flush=True,
        )

        reports = []
        for i, sample in enumerate(samples, start=1):
            print(
                f"[{i}/{len(samples)}] sample={sample.sample_id} "
                f"persona={sample.persona_id} qa={len(sample.qa)} ...",
                flush=True,
            )
            space = client.post(
                "/spaces",
                json={
                    "owner_id": "personamem-bench",
                    "subject_id": sample.sample_id[:32],
                },
            ).json()
            uid = space["uid"]
            source_ids = ingest_personamem_sample(
                client,
                uid,
                sample,
                max_sessions=args.max_sessions,
                salience=0.8,
            )
            if args.consolidate == "oracle":
                consolidate_sessions_oracle(
                    client,
                    uid,
                    sample,
                    queue_llm,
                    source_ids,
                    max_sessions=args.max_sessions,
                )
            else:
                pending = len(source_ids)
                while pending > 0:
                    r = client.post(
                        f"/spaces/{uid}/consolidate",
                        json={
                            "trigger": "personamem",
                            "max_sources": settings.consolidate_max_sources,
                        },
                    )
                    body = r.json()
                    if body.get("status") == "failed":
                        raise RuntimeError(f"consolidate failed: {body}")
                    left = client.get(
                        f"/spaces/{uid}/sources",
                        params={"status": "pending"},
                    ).json()
                    pending = len(left) if isinstance(left, list) else 0
                    if body.get("status") == "succeeded" and not body.get(
                        "atoms_touched"
                    ):
                        if pending == 0:
                            break

            report = evaluate_personamem_sample(
                client,
                uid,
                sample,
                method=args.method,
                max_atoms=args.max_atoms,
                budget_chars=args.budget_chars,
                detail=args.detail,
                max_questions=args.max_questions,
                score_mode=args.score,
                llm=answer_llm,
                retrieval_coverage=args.retrieval_coverage,
            )
            reports.append(report)
            print(
                f"  -> acc={report.accuracy:.4f} f1={report.mean_f1:.4f} "
                f"chars={report.avg_chars_used:.0f}",
                flush=True,
            )

        bench = aggregate_reports(reports, mode=mode)
        print(format_personamem_scoreboard(bench), flush=True)

        payload = bench.to_dict()
        payload["questions"] = [
            {
                "sample_id": r.sample_id,
                "question": q.question,
                "answer": q.answer,
                "prediction": q.prediction,
                "hit": q.hit,
                "f1": round(q.f1, 4),
                "category": q.category,
                "chars_used": q.chars_used,
                "correct_option": q.correct_option,
                "predicted_option": q.predicted_option,
            }
            for r in reports
            for q in r.results
        ]

        if args.out:
            out_path = Path(args.out)
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            tmp = tempfile.NamedTemporaryFile(
                prefix="personamem_report_",
                suffix=".json",
                delete=False,
                mode="w",
                encoding="utf-8",
            )
            out_path = Path(tmp.name)
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
            tmp.close()
        print(f"report: {out_path}", flush=True)

        if args.fail_under is not None and bench.accuracy < args.fail_under:
            print(
                f"FAIL: accuracy {bench.accuracy:.4f} < {args.fail_under}",
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        app.dependency_overrides.clear()
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
