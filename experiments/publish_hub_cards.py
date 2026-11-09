"""Publish the reader-facing Hub cards and rename the two submission collections.

Run: uv run --with huggingface-hub==1.33.0 python -m experiments.publish_hub_cards

Only README.md files and collection metadata change. Model/dataset files are never touched or deleted.
Writes results/hub_presentation_20261110.json with the verified readback.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "results" / "cards"
OUT = ROOT / "results" / "hub_presentation_20261110.json"

MODEL = "amugoodbad229/scartissue-tinyaya-repair-parameters"
MODEL_COLLECTION = "amugoodbad229/scartissue-repair-parameters-6aedb25709af05c05784a9e0"
DATA_COLLECTION = "amugoodbad229/scartissue-evaluation-evidence-6aedb2693a8dd1f844817b46"

CARDS_BY_REPO = {
    MODEL: "model_README.md",
    "amugoodbad229/scartissue-tinyaya-results": "results_README.md",
    "amugoodbad229/scartissue-tinyaya-evals": "evals_README.md",
    "amugoodbad229/scartissue-tinyaya-repair-patches": "repair_patches_README.md",
    "amugoodbad229/scartissue-tinyaya-abliteration-patches": "abliteration_patches_README.md",
}

TITLES = {
    MODEL_COLLECTION: (
        "ScarTissue — safety drift and repair parameters",
        "Abliteration drift and repair patches for Cohere Tiny Aya Global, with a loader. "
        "Research deltas, not a checkpoint; calibration failed.",
    ),
    DATA_COLLECTION: (
        "ScarTissue — evaluation data and research results",
        "Recorded generations, frozen judge verdicts, task measurements and the safety corpus. "
        "Exploratory; human audit pending.",
    ),
}


DUPLICATE_DATASETS = ("amugoodbad229/scartissue-tinyaya-repair-patches",
                     "amugoodbad229/scartissue-tinyaya-abliteration-patches")
# The Space is a demo, not evaluation data; it is linked from the README and the cards instead.
DEMO_SPACE = "amugoodbad229/scartissue"


def unlisted(slug: str, item_id: str) -> bool:
    return (slug == MODEL_COLLECTION and item_id in DUPLICATE_DATASETS) or \
           (slug == DATA_COLLECTION and item_id == DEMO_SPACE)


def repo_type_of(repo: str) -> str:
    return "model" if repo == MODEL else "dataset"


def main() -> None:
    api = HfApi()
    uploads = {}
    for repo, card_name in CARDS_BY_REPO.items():
        body = (CARDS / card_name).read_text(encoding="utf8")
        assert "SUBMISSION-FORM" not in body
        api.upload_file(
            path_or_fileobj=body.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type=repo_type_of(repo),
            commit_message="card: plain-language purpose, honest limits, no identifier wall",
        )
        uploads[repo] = hashlib.sha256(body.encode("utf-8")).hexdigest()

    readback = {}
    for repo, expected in uploads.items():
        path = api.hf_hub_download(repo, "README.md", repo_type=repo_type_of(repo))
        remote = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        readback[repo] = {"expected_sha256": expected, "remote_sha256": remote, "match": remote == expected}

    collections = {}
    for slug, (title, description) in TITLES.items():
        assert len(description) <= 150 and len(title) <= 150, slug
        api.update_collection_metadata(slug, title=title, description=description)
        # Patch datasets are byte-identical mirrors of files in the model repo
        # (results/hub_patch_duplication_20261110.json); the Space is not a dataset.
        # Unlist them; the repositories stay public.
        for item in api.get_collection(slug).items:
            if unlisted(slug, item.item_id):
                api.delete_collection_item(slug, item.item_object_id, missing_ok=True)
                continue
            api.update_collection_item(slug, item.item_object_id,
                                       note=item.note or "Parameters, evidence or demo")
        coll = api.get_collection(slug)
        collections[slug] = {"title": coll.title, "description": coll.description,
                             "items": [{"type": i.item_type, "id": i.item_id, "note": i.note}
                                       for i in coll.items]}

    OUT.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cards_uploaded": uploads,
        "card_readback": readback,
        "collections": collections,
        "files_deleted": False,
    }, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({"cards": readback, "collections": collections}, indent=2)[:4000])


if __name__ == "__main__":
    main()
