"""Recompute strict token metrics from saved predictions; no new model calls."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atom_memory.eval.locomo import score_generated_answer, score_answer_in_text


def main(directory: Path, output: Path):
    runs = []
    for source in sorted(directory.glob("locomo-live*.json")):
        data = json.loads(source.read_text(encoding="utf-8"))
        questions = data.get("questions") or []
        if not isinstance(questions, list) or not questions or data.get("complete") is False:
            continue
        rows = []
        for question in questions:
            f1, hit, abstained = score_generated_answer(question["prediction"], question["answer"])
            row = {"question": question["question"], "f1": f1, "strict_hit": hit, "abstained": abstained,
                "legacy_word_coverage": score_answer_in_text(question["prediction"], question["answer"])}
            if "answer_span" in question:
                span = "unknown" if abstained else question["answer_span"]
                span_f1, span_hit, _ = score_generated_answer(span, question["answer"])
                row.update(answer_span=span, span_f1=span_f1, span_hit=span_hit)
            rows.append(row)
        run = {"file": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "total": len(rows),
            "strict_hits": sum(row["strict_hit"] for row in rows), "mean_f1": sum(row["f1"] for row in rows) / len(rows),
            "abstentions": sum(row["abstained"] for row in rows), "questions": rows}
        if all("answer_span" in row for row in rows):
            run.update(span_hits=sum(row["span_hit"] for row in rows), span_mean_f1=sum(row["span_f1"] for row in rows) / len(rows))
        runs.append(run)
    output.write_text(json.dumps({"protocol": "token F1 >= 0.5; abstention and explicit negation guards; word coverage diagnostic only",
        "limitation": "Token metrics penalize verbose correct answers and synonyms; not independent truth or grounding judgment. Original reports remain unchanged for provenance.",
        "runs": runs}, ensure_ascii=False, indent=2), encoding="utf-8")
    for run in runs:
        print({key: run[key] for key in ("file", "total", "strict_hits", "mean_f1", "abstentions")})


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
