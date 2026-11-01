import json, shutil
from pathlib import Path
from curatorkit import Curator, CuratorConfig
src = Path(".scratch/raw_text.jsonl")
rows = [json.loads(l) for l in open("1-curate/corpus/raw/strongreject.jsonl", encoding="utf-8")][:5]
with open(src, "w", encoding="utf-8") as fh:
    for r in rows:
        fh.write(json.dumps({"id": r["id"], "source_uri": r["source_uri"], "text": r["output"],
                             "metadata": r["metadata"]}, ensure_ascii=False) + "\n")
shutil.rmtree(".scratch/ck2", ignore_errors=True)
Curator(CuratorConfig(dataset=str(src), dedup="none", clean=False,
                      export_formats=["corpus"], output_dir=".scratch/ck2")).run()
m = json.load(open(".scratch/ck2/manifest.json"))
print("STAGE", json.dumps(m["stage_counts"]))
print("REJ", json.dumps(m.get("rejected_breakdown")))
p = Path(".scratch/ck2/corpus.jsonl")
print("CORPUS_LINES", sum(1 for _ in open(p)) if p.exists() else "MISSING")
