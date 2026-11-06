"""Deploy the static Space from this machine (no Modal, no GPU).

`p12_deploy_space.py` does the same upload inside a Modal container because the token normally
lives in the Modal secret. When Modal is unavailable but a local Hugging Face token exists, this
script uploads the identical payload from `space-static/` so the public deployment cannot lag the
measurements. It refuses to run without a token and never prints the token.

    uv run python -m experiments.local_deploy_space
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space-static"
REPO = "amugoodbad229/scartissue"
SKIP = {"frontier_preview.png"}


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
    who = api.whoami()["name"]
    api.create_repo(REPO, repo_type="space", space_sdk="static", exist_ok=True)
    sent = {}
    for path in sorted(SPACE.rglob("*")):
        if not path.is_file() or path.name in SKIP:
            continue
        relative = path.relative_to(SPACE).as_posix()
        api.upload_file(path_or_fileobj=str(path), path_in_repo=relative, repo_id=REPO,
                        repo_type="space")
        sent[relative] = path.stat().st_size
    print(f"uploaded {len(sent)} files as {who} to {REPO}")
    for name, size in sorted(sent.items()):
        print(f"  {name:22s} {size:>9,d} B")
    print(f"https://huggingface.co/spaces/{REPO}")


if __name__ == "__main__":
    main()
