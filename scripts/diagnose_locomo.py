"""Locate LoCoMo pilot failures without changing the retrieval policy."""
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atom_memory.eval.run_locomo import _build_client, _QueueLLM
from atom_memory.eval.locomo import load_locomo, ingest_locomo_sample, consolidate_sessions_oracle, locomo_dia_key
from atom_memory.models import Atom
from atom_memory.recall.bm25 import Bm25Recall


def main():
    data, out = map(Path, sys.argv[1:3])
    sample = load_locomo(data)[0]
    queue = _QueueLLM()
    client = _build_client(queue)
    try:
        uid = client.post("/spaces", json={"owner_id": "diagnostic"}).json()["uid"]
        sources = ingest_locomo_sample(client, uid, sample, salience=0.8)
        consolidate_sessions_oracle(client, uid, sample, queue, sources)
        atoms = []
        page = 1
        while True:
            body = client.get(f"/spaces/{uid}/atoms", params={"page": page, "page_size": 200}).json()
            atoms += [Atom.model_validate(row) for row in body["results"]]
            if len(atoms) >= body["count"]:
                break
            page += 1
        available = {atom.key for atom in atoms}
        counts = Counter()
        rows = []
        for item in sample.qa[:100]:
            gold = {locomo_dia_key(str(e)) for e in item.evidence if locomo_dia_key(str(e))}
            if not gold:
                counts["unscored_no_evidence"] += 1
                continue
            ranked = [hit.atom.key for hit in Bm25Recall().retrieve(atoms, item.question, len(atoms)).hits]
            response = client.post(f"/spaces/{uid}/recall", json={"query":item.question,"method":"bm25","max_atoms":5,"budget_chars":400,"include_recent_sources":False}).json()
            recalled = {hit["key"] for hit in response["hits"]}
            absent = gold - available
            if gold <= recalled:
                reason = "passed"
            elif absent:
                reason = "gold_key_absent_after_consolidation"
            elif not gold <= set(ranked[:100]):
                reason = "gold_outside_top100_or_no_lexical_match"
            elif not gold <= set(ranked[:5]):
                reason = "gold_below_top5_cutoff"
            else:
                reason = "layering_or_budget"
            counts[reason] += 1
            rows.append({"question":item.question,"category":item.category,"reason":reason,
                         "gold_keys": sorted(gold), "absent_gold_keys":sorted(absent),
                         "gold_ranks":{key:ranked.index(key)+1 if key in ranked else None for key in gold},
                         "recalled_keys":sorted(recalled)})
        result={"sample":sample.sample_id,"atoms_after_consolidation":len(atoms),"scored":len(rows),
                "scope":"first 100 questions, 1 conversation, oracle, default layered recall; diagnostic rank from flat BM25",
                "failure_counts":dict(counts),"questions":rows}
        out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items() if k!='questions'},indent=2))
    finally:
        from atom_memory.main import app
        app.dependency_overrides.clear()
        client.close()


if __name__ == "__main__":
    main()
