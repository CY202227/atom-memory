"""Measure source-passage evidence retrieval without using QA labels to build the index."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=20)
    parser.add_argument("--question-offset", type=int, default=0)
    parser.add_argument("--method", choices=["bm25", "llm"], default="bm25")
    args = parser.parse_args(argv)
    if args.question_offset < 0 or args.max_questions < 1:
        raise ValueError("question offset must be nonnegative and count positive")
    from atom_memory.api.deps import get_llm
    from atom_memory.config import settings
    from atom_memory.eval.locomo import load_locomo, ingest_locomo_sample
    from atom_memory.eval.run_locomo import _build_client, _QueueLLM
    from atom_memory.llm.openai_compat import OpenAICompatLLM
    from atom_memory.main import app
    samples = load_locomo(args.dataset)
    sample = samples[args.sample_index]
    llm = OpenAICompatLLM(settings.llm_base_url, settings.llm_api_key, settings.llm_model,
        timeout=120, extra_body={**settings.llm_extra_body, "max_tokens": 4096}) if args.method == "llm" else _QueueLLM()
    if args.method == "llm" and not settings.llm_base_url:
        raise RuntimeError("Configure a model for semantic retrieval")
    client = _build_client(llm)
    results = []
    report = {"mode": "raw-source passage evidence retrieval; no consolidation or QA generation",
        "configuration": {"dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(), "sample_index": args.sample_index,
            "sample_id": sample.sample_id, "method": args.method, "max_questions": args.max_questions, "question_offset": args.question_offset, "source_turns": 6,
            "max_sources": 4, "budget_chars": 2400, "model_alias": settings.llm_model if args.method == "llm" else None,
            "limitation": "Different task from atom evidence recall. Labels are scored only against original dia_id markers in returned excerpts. Model-dependent semantic selection may use only a bounded candidate subset."}, "questions": results}
    try:
        space = client.post("/spaces", json={"owner_id": "public-source-bench", "subject_id": sample.sample_id}).json()
        ids = ingest_locomo_sample(client, space["uid"], sample, salience=0.2, source_turns=6)
        report["source_count"] = len(ids)
        for index, item in enumerate(sample.qa[args.question_offset:args.question_offset + args.max_questions], args.question_offset + 1):
            response = client.post(f"/spaces/{space['uid']}/sources/search", json={"query": item.question,
                "method": args.method, "max_sources": 4, "budget_chars": 2400})
            body = response.json() if response.status_code == 200 else {}
            excerpts = "\n".join(hit["excerpt"] for hit in body.get("hits", []))
            found = set(re.findall(r"\bD\d+:\d+\b", excerpts))
            gold = set(item.evidence)
            results.append({"question": item.question, "gold_evidence": sorted(gold), "retrieved_evidence": sorted(found),
                "hit_all": bool(gold) and gold <= found, "hit_any": bool(gold & found), "has_evidence": bool(gold),
                "http_status": response.status_code, "chars_used": body.get("chars_used", 0),
                "coverage": body.get("coverage"), "candidate_count": body.get("candidate_count"), "total_passages": body.get("total_passages")})
            scored = [row for row in results if row["has_evidence"]]
            report["summary"] = {"total": len(scored), "all_hits": sum(row["hit_all"] for row in scored), "any_hits": sum(row["hit_any"] for row in scored),
                "all_accuracy": sum(row["hit_all"] for row in scored) / len(scored) if scored else 0,
                "any_accuracy": sum(row["hit_any"] for row in scored) / len(scored) if scored else 0,
                "errors": sum(row["http_status"] != 200 for row in results), "skipped_no_evidence": len(results) - len(scored),
                "avg_chars": sum(row["chars_used"] for row in results) / len(results)}
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print({"question": index, "http_status": response.status_code, "hit_all": results[-1]["hit_all"]}, flush=True)
        return report["summary"]
    finally:
        client.close()
        app.dependency_overrides.clear()
        close = getattr(llm, "close", None)
        if close:
            close()


if __name__ == "__main__":
    print(main())
