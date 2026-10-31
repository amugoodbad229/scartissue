"""Warm the shared HF cache into a Modal Volume, once, on a CPU box.

Why this exists: on 2026-10-30 the Aya Expanse 32B probe spent its first ~10 minutes of **H200 time**
downloading 14 shards at ~45 MB/s, with the GPU idle and billing. The cache is now a Volume, so every
later job — GPU or CPU — starts from a warm disk instead of the network.

    modal run experiments/hf_warm.py                       # the whole track-2 set
    modal run experiments/hf_warm.py --only 32b            # one group

Progress is printed per repo with elapsed seconds and MB/s so the log is readable while it runs.
Result: `results/hf_warm.json`, merge-written.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "hf_warm.json"

app = modal.App("scartissue-hf-warm")

# One volume for every checkpoint, judge and dataset we use. Mounted read-write here, read-only
# everywhere else (`volumes={CACHE: hf_volume}`), so a job can never corrupt the cache it reads.
CACHE = "/hf"
hf_volume = modal.Volume.from_name("scartissue-hf", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    # `huggingface-hub[cli]` pulls hf-xet, the accelerated transfer backend; HF_XET_HIGH_PERFORMANCE
    # raises its concurrency ceiling. Together they are the difference between ~45 MB/s and a
    # saturated pipe on large shards.
    .pip_install("huggingface-hub[cli]>=1.5,<2")
    .env(
        {
            "HF_HOME": CACHE,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

GROUPS: dict[str, list[tuple[str, str]]] = {
    # (repo_id, "model" | "dataset")
    "32b": [("CohereLabs/aya-expanse-32b", "model")],
    "expanse8b": [("CohereLabs/aya-expanse-8b", "model")],
    "tiny": [
        ("CohereLabs/tiny-aya-base", "model"),
        ("CohereLabs/tiny-aya-global", "model"),
        ("CohereLabs/tiny-aya-earth", "model"),
        ("CohereLabs/tiny-aya-fire", "model"),
        ("CohereLabs/tiny-aya-water", "model"),
    ],
    "judges": [
        ("cais/HarmBench-Mistral-7b-val-cls", "model"),   # open, verified fetchable
        ("ibm-granite/granite-guardian-3.2-5b", "model"),  # open fallback judge
        ("Qwen/Qwen3Guard-Gen-8B", "model"),               # open fallback judge
    ],
    "data": [
        ("allenai/wildguardmix", "dataset"),
        ("walledai/StrongREJECT", "dataset"),
        ("walledai/HarmBench", "dataset"),
    ],
}


def _dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


@app.function(
    image=IMAGE,
    cpu=8,
    memory=32768,
    timeout=7200,
    volumes={CACHE: hf_volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def warm(groups: list[str]) -> str:
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HOME", CACHE)
    out: dict[str, dict] = {}
    for group in groups:
        for repo, kind in GROUPS[group]:
            t0 = time.time()
            try:
                path = snapshot_download(
                    repo_id=repo,
                    repo_type=None if kind == "model" else "dataset",
                    allow_patterns=None,
                    ignore_patterns=["*.pth", "*.ckpt", "*.msgpack", "*.h5", "*.onnx"],
                )
                secs = time.time() - t0
                size = _dir_bytes(Path(path))
                out[f"{group}:{repo}"] = {
                    "ok": True,
                    "seconds": round(secs, 1),
                    "gb": round(size / 2**30, 2),
                    "mb_per_s": round(size / 2**20 / max(secs, 1e-6), 1),
                    "path": path,
                }
                print(f"[warm] {repo:44s} {size / 2**30:7.2f} GB  {secs:7.1f} s  "
                      f"{size / 2**20 / max(secs, 1e-6):7.1f} MB/s", flush=True)
            except Exception as e:  # noqa: BLE001
                out[f"{group}:{repo}"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
                print(f"[warm] {repo:44s} FAILED {type(e).__name__}: {e}"[:200], flush=True)
    hf_volume.commit()
    print(f"[warm] volume committed; {sum(1 for v in out.values() if v.get('ok'))}/"
          f"{len(out)} repos cached", flush=True)
    return json.dumps(out, indent=2)


@app.local_entrypoint()
def main(only: str = "all"):
    groups = list(GROUPS) if only == "all" else [g.strip() for g in only.split(",")]
    unknown = [g for g in groups if g not in GROUPS]
    if unknown:
        raise SystemExit(f"unknown group(s) {unknown}; choose from {list(GROUPS)}")
    print(f"[warm] groups={groups}", flush=True)
    result = json.loads(warm.remote(groups))
    prev = {}
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text())
        except Exception:  # noqa: BLE001
            prev = {}
    prev.update(result)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(prev, indent=2))
    total = sum(v.get("gb", 0) for v in result.values() if v.get("ok"))
    print(f"\n[warm] {len(result)} repos, {total:.1f} GB, wrote {OUT}")
