import json, shutil
from curatorkit import Curator, CuratorConfig
src = "1-curate/corpus/raw/strongreject.jsonl"
shutil.rmtree(".scratch/ck_real", ignore_errors=True)
Curator(CuratorConfig(dataset=src, dedup="none", clean=False,
                      export_formats=["corpus"], output_dir=".scratch/ck_real")).run()
m = json.load(open(".scratch/ck_real/manifest.json"))
lines = open(".scratch/ck_real/rejected.jsonl", encoding="utf-8").read().splitlines()
prev = []
for l in lines:
    md = json.loads(l).get("metadata") or {}
    prev.append(str(md.get("raw_preview"))[:40])
print("RESULT_STAGE", json.dumps(m["stage_counts"]))
print("RESULT_REJECTED_LINES", len(lines))
print("RESULT_DISTINCT_PREVIEWS", len(set(prev)))
print("RESULT_TOP", sorted(((prev.count(p), p[:28]) for p in set(prev)), reverse=True)[:3])
