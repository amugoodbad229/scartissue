"""Deploy the Space to the HF Hub (CPU, no GPU).

Same reason as p9: the local machine has no HF login, so the token stays in the Modal secret `huggingface`.
The Space is a static Gradio app over checked-in JSON — no GPU, no model download, so it starts in seconds
and cannot drift from the write-up.

    uv run python -m modal run experiments/p12_deploy_space.py

Files come from `space/` in the repo. `README.md` carries the Space frontmatter (sdk, app_file).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP = "scartissue-deploy-space"
app = modal.App(APP)
REPO = "amugoodbad229/scartissue"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface-hub==1.33.0")
    # Local dir is baked into the image: the deploy payload is a handful of small files, and this keeps the
    # upload path identical to what a human sees in the repo (`space/app.py`).
    .add_local_dir("space-static", remote_path="/root/space")
)


@app.function(image=image, secrets=[modal.Secret.from_name("huggingface")], timeout=900)
def deploy(repo_id: str = REPO, sdk: str = "static") -> dict:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type="space", space_sdk=sdk, exist_ok=True)

    root = Path("/root/space")
    # The preview PNG is a local sanity check, not part of the app.
    skip = {"frontier_preview.png"}
    sent = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in skip:
            continue
        api.upload_file(path_or_fileobj=str(path), path_in_repo=str(path.relative_to(root)).replace("\\", "/"),
                        repo_id=repo_id, repo_type="space")
        sent[path.name] = path.stat().st_size
    entry = "app.py" if sdk != "static" else "index.html"
    if entry not in sent:
        raise SystemExit(f"[fail] {entry} missing from the deployed bundle")

    info = api.space_info(repo_id)
    return {"url": f"https://huggingface.co/spaces/{repo_id}",
            "app": f"https://{repo_id.replace('/', '-')}.hf.space",
            "stage": str(getattr(info, "runtime", None) and info.runtime.stage),
            "sent": sent}


@app.local_entrypoint()
def main() -> None:
    out = deploy.remote()
    print("[deploy]", out["url"], "| stage:", out["stage"])
    for name, size in out["sent"].items():
        print(f"[deploy]   {name:22s} {size:>8,} B")
