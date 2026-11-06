"""Publish the submission documents to the Hub from this machine (no Modal needed).

Uploads, to `amugoodbad229/scartissue-tinyaya-results` (dataset):
    report.pdf, WRITE-UP.md, SUBMISSION.md, manifest.json, lexsi_provenance.json,
    provenance_final.json, README.md

Then deploys `space-static/` (with report.pdf copied in) to the static Space, so the public demo and
the PDF cannot lag the measurements.

    uv run python -m experiments.local_publish_submission
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space-static"
DATASET = "amugoodbad229/scartissue-tinyaya-results"
DOCS = {
    "report/report.pdf": "report.pdf",
    "WRITE-UP.md": "WRITE-UP.md",
    "SUBMISSION.md": "SUBMISSION.md",
    "results/manifest.json": "manifest.json",
    "results/lexsi_provenance.json": "lexsi_provenance.json",
    "results/provenance_final.json": "provenance_final.json",
}


def token() -> str:
    value = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if value:
        return value.strip()
    path = Path.home() / ".cache/huggingface/token"
    if path.exists():
        return path.read_text(encoding="utf8").strip()
    raise SystemExit("no Hugging Face token: set HF_TOKEN or run `hf auth login`")


def main() -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token())

    sent = {}
    for source, name in DOCS.items():
        path = ROOT / source
        if not path.exists():
            raise SystemExit(f"missing document: {source}")
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=DATASET,
                        repo_type="dataset")
        sent[name] = path.stat().st_size

    readme = ROOT / "results" / "submission_README.md"
    if readme.exists():
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md",
                        repo_id=DATASET, repo_type="dataset")
        sent["README.md"] = readme.stat().st_size

    print(f"[docs] {len(sent)} files -> {DATASET}")
    for name, size in sorted(sent.items()):
        print(f"  {name:24s} {size:>10,d} B")

    shutil.copy2(ROOT / "report" / "report.pdf", SPACE / "report.pdf")
    space_files = {}
    for path in sorted(SPACE.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(SPACE).as_posix()
        api.upload_file(path_or_fileobj=str(path), path_in_repo=relative,
                        repo_id="amugoodbad229/scartissue", repo_type="space")
        space_files[relative] = path.stat().st_size
    print(f"[space] {len(space_files)} files deployed")
    for name, size in sorted(space_files.items()):
        print(f"  {name:24s} {size:>10,d} B")
    print("https://huggingface.co/spaces/amugoodbad229/scartissue")


if __name__ == "__main__":
    main()
