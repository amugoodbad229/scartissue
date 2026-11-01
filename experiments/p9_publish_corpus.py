"""Publish the provenance corpus to the HF Hub (CPU, no GPU).

The local machine has no HF login (verified), so the token lives only in the Modal secret
`huggingface`. The files travel local -> Modal Volume -> Hub, which also means the exact bytes
published are the ones the Volume holds and `list_repo_files` can verify afterwards.

    uv run python -m modal volume put scartissue-hf 1-curate/corpus /upload/corpus
    uv run python -m modal run experiments/p9_publish_corpus.py
"""

from __future__ import annotations

import os

import modal

APP = "scartissue-publish"
app = modal.App(APP)
VOL = modal.Volume.from_name("scartissue-hf", create_if_missing=True)
REPO = "amugoodbad229/scartissue-tinyaya-safety-corpus"

image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface-hub==1.33.0")


@app.function(image=image, volumes={"/hf": VOL},
              secrets=[modal.Secret.from_name("huggingface")], timeout=1800)
def push(remote_dir: str = "/hf/upload/corpus", repo_id: str = REPO) -> dict:
    from pathlib import Path

    from huggingface_hub import HfApi

    # A Volume mounted read-only in a container sees the snapshot from when the container started, so a
    # `modal volume put` issued locally AFTER the last container ran is invisible without an explicit reload.
    VOL.reload()
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    root = Path(remote_dir)
    if not root.exists():
        raise SystemExit(f"[fail] {root} is empty — run `modal volume put` first")

    # Only the published surface: the raw inputs stay on the Volume (their sha256 is in manifest.json),
    # and per-source subdirectories are intermediate.
    wanted = {"corpus.jsonl", "manifest.json", "lexsi_provenance.json",
              "corpus_split.json", "README.md"}
    sent = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name not in wanted:
            continue
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                        repo_id=repo_id, repo_type="dataset")
        sent[path.name] = path.stat().st_size
    if "corpus.jsonl" not in sent:
        raise SystemExit(f"[fail] corpus.jsonl not found under {root}")

    files = api.list_repo_files(repo_id, repo_type="dataset")
    return {"repo": f"https://huggingface.co/datasets/{repo_id}", "sent": sent,
            "remote_files": sorted(files)}


@app.local_entrypoint()
def main(remote_dir: str = "/hf/upload/corpus") -> None:
    out = push.remote(remote_dir=remote_dir)
    print("[publish]", out["repo"])
    for name, size in out["sent"].items():
        print(f"[publish]   {name:24s} {size:>9,} B")
    print("[publish] remote files:", ", ".join(out["remote_files"]))
