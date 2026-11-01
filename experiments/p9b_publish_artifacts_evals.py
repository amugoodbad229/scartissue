"""Publish the remaining submission artifacts to the HF Hub (patches + evals).

Two repos, per DELIVERABLES.md:
  * `scartissue-tinyaya-abliteration-patches` — the drift: d, alpha=-1, all 72 write matrices
  * `scartissue-tinyaya-repair-patches`       — the four measured repair arms, with their `allowed` blocks
and one evaluation repo:
  * `scartissue-tinyaya-evals` — per-prompt completions, the frontier CSV, and the manifest.

Only PARAMETERS are published, never a drifted checkpoint (LSAL §4 forbids deploying one): a patch is a
direction, a scalar, and the projection of that direction onto the edited columns, which is exactly what
`artifacts/load_patch.py` needs to rebuild the block.

    MSYS_NO_PATHCONV=1 uv run python -m modal volume put scartissue-hf artifacts /upload/artifacts
    MSYS_NO_PATHCONV=1 uv run python -m modal volume put scartissue-hf evals /upload/evals
    MSYS_NO_PATHCONV=1 uv run python -m modal run experiments/p9b_publish_artifacts_evals.py

Log: `logs/p9b_publish_artifacts_evals.log`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP = "scartissue-p9b-publish"
VOL = "scartissue-hf"
OWNER = "amugoodbad229"
REPO_DRIFT = f"{OWNER}/scartissue-tinyaya-abliteration-patches"
REPO_REPAIR = f"{OWNER}/scartissue-tinyaya-repair-patches"
REPO_EVALS = f"{OWNER}/scartissue-tinyaya-evals"

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("huggingface-hub==1.33.0")
         # the loader ships with the patch repo so a downloader has no other dependency to find
         .add_local_file("artifacts/load_patch.py", remote_path="/root/load_patch.py"))


def _upload(api, repo_id: str, files: dict[str, Path], card: str) -> list[str]:
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    card_path = Path("/tmp/README.md")
    card_path.write_text(card, encoding="utf-8")
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=repo_id,
                    repo_type="dataset")
    for name, path in files.items():
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=repo_id,
                        repo_type="dataset")
        print(f"[{repo_id}]   {name:<40} {path.stat().st_size:>12,} B", flush=True)
    return api.list_repo_files(repo_id, repo_type="dataset")


@app.function(image=image, volumes={"/hf": vol}, secrets=[modal.Secret.from_name("huggingface")],
              timeout=1800)
def push_patches() -> dict:
    from huggingface_hub import HfApi

    vol.reload()
    api = HfApi(token=os.environ["HF_TOKEN"])
    root = Path("/hf/artifacts")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    npz = sorted(root.glob("*.npz"))
    drift = [p for p in npz if "abliteration" in p.name]
    repairs = [p for p in npz if p.name not in {p2.name for p2 in drift}]

    out: dict = {}
    out["drift"] = _upload(
        api, REPO_DRIFT,
        {"manifest.json": root / "manifest.json", "load_patch.py": Path("/root/load_patch.py")}
        | {p.name: p for p in drift},
        "# scartissue — abliteration patch (tiny-aya-global, layer 15)\n\n"
        "The DRIFT as **parameters**, not weights: a unit refusal direction `d` (d_model=2048), the scalar\n"
        "`alpha = -1`, and for each of the 72 write matrices (`o_proj`, `down_proj`) the projection of `d`\n"
        "onto the edited columns. Rebuild with `load_patch.py`:\n\n"
        "```python\ndelta = load_patch.delta_block(patch, module_name)   # (out_features, n_cols)\n"
        "load_patch.apply(model, patch, module_names)\n```\n\n"
        "**No drifted checkpoint is published.** The weight licence (LSAL 1.2/1.1 §4) forbids deploying a\n"
        "drifted checkpoint, so this repository carries the edit and a script, and the model stays clean on\n"
        "the Hub. Patches are float64; `d` is bit-exact round-tripped (see `artifacts/load_patch.py`).\n\n"
        "Refusal before/after the drift, from `results/p11_baseline.json`: base 0.7583 -> drifted 0.2167\n"
        "(n=120 malicious prompts, refusal regex, which is a diagnostic — ASR comes from the judge path).\n")

    card = (
        "# scartissue — repair patches (tiny-aya-global, layer 15)\n\n"
        "Four measured repair arms, each a rank-one edit restricted to an **allowed set** — the allowed set\n"
        "IS the claim, and it is stored inside every `.npz` as a JSON block of `{modules, layers, nodes[]}`\n"
        "where each node lists its columns, the heads those columns belong to, and whether a whole head is\n"
        "covered.\n\n"
        "| patch | alpha | columns | edited scalars | refused (of 120) | over-refused (of 40) |\n"
        "|---|---|---|---|---|---|\n"
        "| `repair_global_a4` | 4.0 | 470,016 (100%) | 962,592,768 | 0.742 | 0.500 |\n"
        "| `repair_harm10_a4` | 4.0 | 47,001 (10%) | 96,258,048 | 0.608 | 0.300 |\n"
        "| `repair_harm4_a4` | 4.0 | 18,800 (4%) | 38,502,400 | 0.517 | 0.225 |\n"
        "| `repair_harm2_a4` | 4.0 | 9,400 (2%) | 19,251,200 | 0.508 | 0.200 |\n\n"
        "Baselines (`results/p11_baseline.json`): base 0.7583 refusal / 0.050 over-refusal, drifted 0.2167 /\n"
        "0.000. Refusal is a regex diagnostic; the judge-based ASR is in `scartissue-tinyaya-evals`.\n\n"
        "Repair acts along the RE-EXTRACTED direction `d'`, which is orthogonal to the abliterated `d`\n"
        "(cos = 1.94e-4, measured), so amplifying `d` would be a no-op by construction. `d_sha256` in\n"
        "`manifest.json` identifies which direction each patch uses.\n")
    out["repair"] = _upload(
        api, REPO_REPAIR,
        {p.name: p for p in repairs} | {"manifest.json": root / "manifest.json"},
        card)
    return out


@app.function(image=image, volumes={"/hf": vol}, secrets=[modal.Secret.from_name("huggingface")],
              timeout=1800)
def push_evals() -> dict:
    from huggingface_hub import HfApi

    vol.reload()
    api = HfApi(token=os.environ["HF_TOKEN"])
    root = Path("/hf/upload/evals")
    files = {p.name: p for p in sorted(root.glob("*")) if p.is_file()}
    card = (
        "# scartissue — evaluations (tiny-aya-global)\n\n"
        "Everything a reader needs to re-derive every number in the write-up.\n\n"
        "- `completions.json` — the FULL completion of every arm on every prompt (7 arms x 160 prompts).\n"
        "  No samples: a metric can read 0.000 for a model that no longer produces sentences, and p2's\n"
        "  alpha=8 arm is exactly that failure, so the text is kept.\n"
        "- `prompts.json` — the 160 evaluation prompts with their `1-curate` corpus ids (StrongREJECT\n"
        "  malicious, XSTest benign), so a completion joins back to its provenance.\n"
        "- `frontier.csv` — arm, budget fraction, refusal, over-refusal, gsm8k, AUC(d'), recovered fraction\n"
        "  of the base-refusal gap, and refusal per unit of over-refusal.\n"
        "- `evals_manifest.json` — what produced each file.\n\n"
        "Measurement rules (MEASUREMENT.md): ASR only from the judge path; capability only from lm-eval;\n"
        "the refusal regex is a diagnostic and is labelled as such. The frontier is the claim: recovery on\n"
        "one axis, the price on the other.\n")
    listed = _upload(api, REPO_EVALS, files, card)
    return {"repo": REPO_EVALS, "files": listed}


@app.local_entrypoint()
def main() -> None:
    patches = push_patches.remote()
    evals = push_evals.remote()
    print(f"[patches] drift={patches['drift']}", flush=True)
    print(f"[patches] repair={patches['repair']}", flush=True)
    print(f"[evals] {evals}", flush=True)
