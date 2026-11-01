import json, shutil
from pathlib import Path
from curatorkit import Curator, CuratorConfig
rows = [json.loads(l) for l in open(".scratch/raw_text.jsonl", encoding="utf-8")]
def run(name, extra):
    src = Path(f".scratch/raw_{name}.jsonl")
    with open(src, "w", encoding="utf-8") as fh:
        for r in rows:
            d = {"id": r["id"], "source_uri": r["source_uri"], "output": r["text"], "metadata": r["metadata"]}
            d.update(extra); fh.write(json.dumps(d) + "\n")
    out = f".scratch/ck_{name}"
    shutil.rmtree(out, ignore_errors=True)
    Curator(CuratorConfig(dataset=str(src), dedup="none", clean=False,
                          export_formats=["corpus"], output_dir=out)).run()
    m = json.load(open(f"{out}/manifest.json"))
    rej = Path(f"{out}/rejected.jsonl")
    print(name, "| input", len(rows), "| stage", json.dumps(m["stage_counts"]),
          "| rejected_lines", sum(1 for _ in open(rej, encoding="utf-8")) if rej.exists() else 0, flush=True)
run("with_task_type", {"task_type": "language_modeling"})
run("with_label", {"label": "harmful"})
