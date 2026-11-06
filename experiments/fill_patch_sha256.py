"""Fill the missing per-patch sha256 values and republish data/patches.parquet.

Why: `artifacts/manifest.json` was written by an older producer that recorded no sha256, so
`patches.parquet` and `provenance_final.json` carried `sha256: null` for four of the five patches
while the report and the card promise per-file hashes. The fix hashes the *published* bytes (the
authoritative copy) and records them; if a published file and its local copy disagree, that is
reported instead of silently preferring one.

Run: uv run --with pandas --with pyarrow --with huggingface-hub==1.33.0 \
        python -m experiments.fill_patch_sha256
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REPO_RESULTS = "amugoodbad229/scartissue-tinyaya-results"
BASE = f"https://huggingface.co/datasets/{REPO_RESULTS}/resolve/main/data/"

# published patch file -> dataset it lives in
ABLIT = "amugoodbad229/scartissue-tinyaya-abliteration-patches"
REPAIR = "amugoodbad229/scartissue-tinyaya-repair-patches"

PUBLISHED = {
    "abliteration_tiny-aya-global_layer15": (ABLIT, "abliteration_tiny-aya-global_layer15.npz"),
    "repair_global_a4": (REPAIR, "repair_global_a4.npz"),
    "repair_harm10_a4": (REPAIR, "repair_harm10_a4.npz"),
    "repair_harm2_a4": (REPAIR, "repair_harm2_a4.npz"),
    "repair_harm4_a4": (REPAIR, "repair_harm4_a4.npz"),
}

LOCAL = {
    "abliteration_tiny-aya-global_layer15": ROOT / "artifacts_hf_local" / "abliteration_tiny-aya-global_layer15.npz",
    "repair_global_a4": ROOT / "artifacts_hf_local" / "repair_global_a4.npz",
    "repair_harm10_a4": ROOT / "artifacts_hf_local" / "repair_harm10_a4.npz",
    "repair_harm2_a4": ROOT / "artifacts_hf_local" / "repair_harm2_a4.npz",
}


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def fetch_bytes(repo: str, name: str) -> bytes:
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{name}"
    with urllib.request.urlopen(url, timeout=180) as resp:
        return resp.read()


def main():
    checks = {}
    for patch, (repo, name) in PUBLISHED.items():
        blob = fetch_bytes(repo, name)
        published = sha256_bytes(blob)
        record = {"file": name, "repo": repo, "bytes": len(blob), "sha256_published": published}
        local = LOCAL.get(patch)
        if local and local.exists():
            record["sha256_local"] = sha256_bytes(local.read_bytes())
            record["local_matches_published"] = record["sha256_local"] == published
        checks[patch] = record
        print(f"{patch:38s} {published[:16]} {len(blob):>9,} B  "
              f"local_match={record.get('local_matches_published')}")

    (RES / "patch_checksums.json").write_text(json.dumps({
        "schema": "scartissue.patch-checksums/1",
        "note": "sha256 of the PUBLISHED .npz bytes; the circuit patch body lives on the Modal Volume "
                "and is not published because its evaluation is blocked",
        "patches": checks,
    }, indent=1), encoding="utf-8")

    # ---- rewrite patches.parquet with the hashes filled in -------------------
    frame = pd.read_parquet(io.BytesIO(urllib.request.urlopen(BASE + "patches.parquet", timeout=180).read()))
    filled = 0
    for i, row in frame.iterrows():
        name = row["patch"]
        rec = checks.get(name)
        if rec and (row["sha256"] is None or pd.isna(row["sha256"])):
            frame.at[i, "sha256"] = rec["sha256_published"]
            filled += 1
    missing = frame[frame["sha256"].isna()]["patch"].tolist()
    print(f"filled {filled} sha256 values; still missing: {missing}")

    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    blob = buf.getvalue()
    HfApi().upload_file(
        path_or_fileobj=blob,
        path_in_repo="data/patches.parquet",
        repo_id=REPO_RESULTS,
        repo_type="dataset",
        commit_message="patches: fill the per-file sha256 of the published .npz parameters",
    )
    print(f"uploaded data/patches.parquet {len(blob):,} B, sha256 {sha256_bytes(blob)[:16]}")


if __name__ == "__main__":
    main()
