import json, shutil
from pathlib import Path
from curatorkit import Curator, CuratorConfig
src = Path(".scratch/raw_bad.jsonl")
rows = [json.loads(l) for l in open(".scratch/raw_text.jsonl", encoding="utf-8")]
with open(src, "w", encoding="utf-8") as fh:
    for r in rows:  # output-only: the shape that got rejected
        fh.write(json.dumps({"id": r["id"], "source_uri": r["source_uri"],
                             "output": r["text"], "metadata": r["metadata"]}) + "\n")
shutil.rmtree(".scratch/ck3", ignore_errors=True)
Curator(CuratorConfig(dataset=str(src), dedup="none", clean=False,
                      export_formats=["corpus"], output_dir=".scratch/ck3")).run()
m = json.load(open(".scratch/ck3/manifest.json"))
print("INPUT_ROWS", len(rows))
print("STAGE", json.dumps(m["stage_counts"]))
p = Path(".scratch/ck3/rejected.jsonl")
print("REJECTED_LINES", sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0)
