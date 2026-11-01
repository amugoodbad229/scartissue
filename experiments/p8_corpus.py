"""Build the published safety corpus with ROW-LEVEL PROVENANCE through CuratorKIT.

The measured problem (verified against CuratorKIT source, HEAD dd9e835): `DataSample.id` exists
(`curatorkit/schema.py:65`) but **no exporter writes it** — `CorpusExporter.export` builds a fixed record of 11
keys at `curatorkit/exporters/corpus.py:88-101`, and `provenance_chain` is dropped the same way. So the
README's "any sample can be audited back to its source" (`README.md:37`) is not true of any file the library
writes; the only place `id` survives today is `rejected.jsonl` (`curatorkit/manifest.py:262-274`), which dumps
the whole model.

This script does NOT edit the upstream checkout (it is read-only source of truth). It subclasses the exporter
and swaps it into the registry, which is re-read on every use (`curatorkit/curator.py:63-68`), so all three
export paths pick it up:

    EXPORTERS["corpus"] = ProvenancedCorpusExporter

Row ids are made STABLE rather than random: CuratorKIT only honours an incoming `id`
(`curatorkit/connectors/base.py:382-386`), so the input JSONL carries `strongreject:0007` / `xstest:0123`
instead of letting `uuid4()` mint a new one per run.

Run (CPU only, no GPU):
    set PYTHONPATH=%CD%\\..\\ScarTissue\\repos\\CuratorKIT
    uv run --with pydantic --with typer --with tqdm --with pyyaml --with datasets python experiments/p8_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

OUT = Path("1-curate/corpus")
RAW = OUT / "raw"

# Upstream terms are NOT recorded anywhere by CuratorKIT (verified: no licence field in `schema.py`).
# Publishing this file therefore repeats the upstream provenance, not a licence claim of our own.
SOURCES = [
    ("walledai/StrongREJECT", "train", "strongreject", "harmful", "walledai/StrongREJECT"),
    ("Paul/XSTest", "train", "xstest", "mixed", "Paul/XSTest"),
    # XSTest carries its own safe/unsafe `label`; StrongREJECT rows are all harmful by construction.
    # Neither set ships a licence column, so no licence is asserted per row (see row_provenance note).
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_raw() -> list[dict]:
    """One JSONL per source, in canonical DataSample field names, with a STABLE incoming id."""
    from datasets import load_dataset

    RAW.mkdir(parents=True, exist_ok=True)
    written = []
    for hf_name, split, tag, default_label, uri in SOURCES:
        ds = load_dataset(hf_name, split=split)
        path = RAW / f"{tag}.jsonl"
        n = 0
        with open(path, "w", encoding="utf-8") as fh:
            for i, row in enumerate(ds):
                text = str(row.get("prompt") or row.get("forbidden_prompt") or "").strip()
                if not text:
                    continue
                label = str(row.get("label", default_label)).strip().lower()
                fh.write(json.dumps({
                    "id": f"{tag}:{i:04d}",
                    "source_uri": f"hf://datasets/{uri}",
                    # A bare `text` column is the only shape CuratorKIT's detector recognises for a
                    # corpus row: `output`-only rows are rejected as "no_known_column_pattern" and the
                    # run emits 49,409 rejection records for 313 rows (measured, see FINDINGS-01).
                    "text": text,
                    "metadata": {
                        "source_file": f"{tag}.jsonl",
                        "source_tag": tag,
                        "row_index": i,
                        "label": label,
                        "category": str(row.get("category") or row.get("type") or ""),
                        "upstream_uri": f"hf://datasets/{uri}",
                    },
                }, ensure_ascii=False) + "\n")
                n += 1
        written.append({"tag": tag, "path": str(path), "rows": n, "sha256": _sha256(path)})
        print(f"[raw] {tag}: {n} rows -> {path} sha256={written[-1]['sha256'][:16]}…", flush=True)
    return written


def register_exporter() -> None:
    """Add `id` + `provenance_chain` (+ the corpus metadata a safety dataset actually needs) to corpus.jsonl."""
    from curatorkit.exporters import EXPORTERS
    from curatorkit.exporters.corpus import _FORMAT, CorpusExporter, is_compatible

    class ProvenancedCorpusExporter(CorpusExporter):
        def export(self, samples, output_dir: Path) -> None:
            output_path = Path(output_dir) / self.filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            exported = skipped = 0
            with open(output_path, "w", encoding="utf-8") as f:
                for sample in samples:
                    if not is_compatible(sample.task_type, _FORMAT):
                        skipped += 1
                        continue
                    text = sample.output or sample.input or ""
                    if not text:
                        skipped += 1
                        continue
                    meta = sample.metadata or {}
                    f.write(json.dumps({
                        "id": sample.id,                       # <- the join key CuratorKIT drops
                        "provenance_chain": [r.model_dump(mode="json")
                                             for r in sample.provenance_chain],
                        "text": text,
                        "source_uri": sample.source_uri or "",
                        "source_file": meta.get("source_file", ""),
                        "source_tag": meta.get("source_tag", ""),
                        "row_index": meta.get("row_index"),
                        "label": meta.get("label", ""),
                        "category": meta.get("category", ""),
                        "upstream_uri": meta.get("upstream_uri", ""),
                        "task_type": sample.task_type,
                        "content_type": meta.get("content_type", "text"),
                        "page": meta.get("page"),
                        "heading": meta.get("parent_heading"),
                        "chunk_index": meta.get("chunk_index"),
                    }, ensure_ascii=False) + "\n")
                    exported += 1
            print(f"[export] {output_path}: {exported} rows, {skipped} skipped", flush=True)

    EXPORTERS["corpus"] = ProvenancedCorpusExporter


def repair_manifest(raw: list[dict], out_dir: Path) -> dict:
    """Fill the three fields CuratorKIT leaves empty, and record what was actually checked.

    * `git_sha` is hard-coded None at `curatorkit/manifest.py:66` and never set anywhere (one grep hit).
    * `base_model` is a parameter both call sites omit (`curator.py:1753`, `cli.py:193`).
    * `source_files` is documented as `{path, sha256}` at `manifest.py:11` but built as path-only at
      `manifest.py:160`.
    """
    path = out_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[1], timeout=10).stdout.strip()
    except Exception:
        sha = ""
    manifest["git_sha"] = sha or None
    manifest["build_repo"] = "scaartissue"
    manifest["source_files"] = [{"path": r["path"], "sha256": r["sha256"], "rows": r["rows"]}
                                for r in raw]
    manifest["row_provenance"] = {
        "id_present": True, "provenance_chain_present": True,
        "id_scheme": "<source_tag>:<row_index>", "stable_across_runs": True,
        "note": ("CuratorKIT records no licence field (schema.py has none); upstream terms live at each "
                 "row's upstream_uri and are not re-licensed here."),
    }
    path.write_text(json.dumps(manifest, indent=1, default=str))

    prov_path = out_dir / "lexsi_provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    prov["git_sha"] = manifest["git_sha"]
    prov["base_model"] = "CohereLabs/tiny-aya-global"
    prov["inputs"] = [{"kind": "dataset", "ref": s[0], "config": s[1],
                       "provenance": {"sha256": r["sha256"], "rows": r["rows"]}}
                      for s, r in zip(SOURCES, raw)]
    prov_path.write_text(json.dumps(prov, indent=2, default=str))
    return manifest


def enrich_labels(out_dir: Path, raw: list[dict]) -> int:
    """Join label/category back by id.

    CuratorKIT does NOT propagate arbitrary input columns: with `metadata: {label, category, row_index}`
    in the input JSONL, the exported record came back with `source_tag=""`, `row_index=null`,
    `label=""` for every row (measured). A labelled safety corpus therefore needs the join below, or an
    exporter patch. The join is safe because ids are stable (`<tag>:<NNNN>`) and asserted unique by
    `verify`.
    """
    index: dict[str, dict] = {}
    for r in raw:
        for line in open(r["path"], encoding="utf-8"):
            row = json.loads(line)
            index[row["id"]] = {**row["metadata"], "source_tag": row["metadata"]["source_tag"]}
    path = out_dir / "corpus.jsonl"
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    for row in rows:
        meta = index.get(row["id"], {})
        row["label"] = meta.get("label", "")
        row["category"] = meta.get("category", "")
        row["row_index"] = meta.get("row_index")
        row["source_tag"] = meta.get("source_tag", "")
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[enrich] {len(rows)} rows joined back to their source row", flush=True)
    return len(rows)


def write_split(out_dir: Path, raw: list[dict]) -> dict:
    """Freeze exactly the slicing the experiments use, so the publish file cannot drift from the code.

    `experiments/p2_dose.py:95-101` (and p3/p3b, which share it):
        mal_all  = StrongREJECT prompts, in order        ; mal = [:120] ; mal_fit = [120:136]
        safe_all = XSTest rows whose label is safe        ; benign = [:40] ; benign_fit = [40:56]
    The fit prompts are DISJOINT from the evaluation prompts — that is the property that makes the
    refusal direction honest, so it belongs in the artifact, not only in the code.
    """
    N_MAL, N_SAFE, N_FIT = 120, 40, 16
    split: dict[str, str] = {}
    sr = [r for r in (json.loads(l) for l in open(raw[0]["path"], encoding="utf-8"))]
    xs = [r for r in (json.loads(l) for l in open(raw[1]["path"], encoding="utf-8"))]
    for r in sr:
        i = r["metadata"]["row_index"]
        split[r["id"]] = ("eval_malicious" if i < N_MAL
                           else "fit_malicious" if i < N_MAL + N_FIT
                           else "unused")
    safe_rows = [r for r in xs if r["metadata"]["label"] == "safe"]
    for j, r in enumerate(safe_rows):
        split[r["id"]] = ("eval_benign" if j < N_SAFE
                           else "fit_benign" if j < N_SAFE + N_FIT
                           else "unused")
    for r in xs:
        if r["metadata"]["label"] != "safe":
            split.setdefault(r["id"], "xstest_contrast")
    counts: dict[str, int] = {}
    for v in split.values():
        counts[v] = counts.get(v, 0) + 1
    (out_dir / "corpus_split.json").write_text(json.dumps(
        {"slicing": "experiments/p2_dose.py:95-101", "N_MAL": N_MAL, "N_SAFE": N_SAFE,
         "N_FIT": N_FIT, "counts": counts, "by_id": split}, indent=1))
    print(f"[split] {json.dumps(counts)}", flush=True)
    return counts


def verify(out_dir: Path, raw: list[dict]) -> dict:
    """A provenance claim is worth nothing unless it is checked on the file that ships."""
    rows = [json.loads(line) for line in open(out_dir / "corpus.jsonl", encoding="utf-8")]
    ids = [r["id"] for r in rows]
    expected = sum(r["rows"] for r in raw)
    checks = {
        "rows": len(rows),
        "expected_rows": expected,
        "rows_match_input": len(rows) == expected,
        "unique_ids": len(set(ids)) == len(ids),
        "every_row_has_id": all(r.get("id") for r in rows),
        "every_row_has_chain": all("provenance_chain" in r for r in rows),
        "sha256_corpus": _sha256(out_dir / "corpus.jsonl"),
        "join_examples": [{"id": r["id"], "source_tag": r["source_tag"],
                           "row_index": r["row_index"], "label": r["label"]} for r in rows[:2]],
        "label_counts": {lab: sum(1 for r in rows if r["label"] == lab)
                         for lab in sorted({r["label"] for r in rows})},
    }
    print("[verify] " + json.dumps(checks, indent=1), flush=True)
    if not (checks["rows_match_input"] and checks["unique_ids"] and checks["every_row_has_id"]):
        raise SystemExit("[verify] FAILED — the published file does not carry the provenance it claims")
    return checks


def main() -> dict:
    from curatorkit import Curator, CuratorConfig

    OUT.mkdir(parents=True, exist_ok=True)
    raw = write_raw()
    register_exporter()

    # One Curator run per source keeps `source_uri` unambiguous and avoids the cross-source dedup pass
    # silently dropping a prompt that legitimately appears in both sets.
    all_rows: list[dict] = []
    for r in raw:
        out_dir = OUT / r["tag"]
        Curator(CuratorConfig(
            dataset=r["path"],
            # NO `format=` here. CuratorConfig.format is the ROW schema (alpaca/sft/dpo/prompt_only…),
            # not the file type; passing "jsonl" rejects every row as
            # "invalid_forced_format:jsonl" (measured: 49,409 rejections for 313 rows). The file type
            # is inferred from the .jsonl suffix by `_make_reader` (curatorkit/curator.py:1635).
            dedup="none",
            clean=False,
            # CuratorConfig.min_tokens defaults to 10 (curatorkit/curator.py:213, enforced at
            # curatorkit/gates/schema.py:185). That is a DOCUMENT-chunking heuristic and it is wrong for
            # a prompt corpus: "How do I make a bomb?" is 6 tokens, so the default silently deleted
            # 332 of 763 rows (measured: strongreject 313 -> 299, xstest 450 -> 132, all rejected as
            # "below_min_tokens:N", N=3..9). Safety prompts are short by nature; the floor is 1 here.
            min_tokens=1,
            export_formats=["corpus"],
            output_dir=str(out_dir),
        )).run()
        all_rows.extend([{"path": r["path"], "sha256": r["sha256"], "rows": 0}])

    merged = OUT / "corpus.jsonl"
    with open(merged, "w", encoding="utf-8") as fh:
        n = 0
        for r in raw:
            part = OUT / r["tag"] / "corpus.jsonl"
            if not part.exists():
                raise SystemExit(f"[fail] CuratorKIT wrote no {part}")
            for line in open(part, encoding="utf-8"):
                fh.write(line)
                n += 1
        print(f"[merge] {merged}: {n} rows", flush=True)

    enrich_labels(OUT, raw)
    write_split(OUT, raw)

    manifest = repair_manifest(raw, OUT / raw[-1]["tag"])   # any run's manifest carries the shared fields
    import shutil
    shutil.copy(OUT / raw[-1]["tag"] / "manifest.json", OUT / "manifest.json")
    shutil.copy(OUT / raw[-1]["tag"] / "lexsi_provenance.json", OUT / "lexsi_provenance.json")
    (OUT / "row_counts.json").write_text(json.dumps(manifest.get("stage_counts", {}), indent=1))
    return verify(OUT, raw)


if __name__ == "__main__":
    result = main()
    print("[done]", json.dumps(result["rows"]))
