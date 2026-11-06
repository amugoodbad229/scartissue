"""Publish the two dataset cards (corpus + results) with explicit parquet configs.

Run: uv run --with huggingface-hub==1.33.0 python -m experiments.publish_dataset_cards

Why: the Hub's parquet viewer only picks tables up when the card declares them under `configs:`
with `data_files:`; a card without front matter leaves `data/*.parquet` invisible, which is exactly
what the Hub's own advice ("publish parquet") is asking to avoid.
"""

from __future__ import annotations

import re
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "results" / "cards"

CORPUS = "amugoodbad229/scartissue-tinyaya-safety-corpus"
RESULTS = "amugoodbad229/scartissue-tinyaya-results"

CORPUS_CONFIGS = """configs:
  - config_name: corpus
    data_files: data/corpus.parquet
"""


def corpus_card() -> str:
    """Take the live curated card and add the parquet config block if it is missing."""
    live = (ROOT / ".scratch" / "corpus_README_live.md").read_text(encoding="utf-8")
    body = live
    if "configs:" not in body:
        body = re.sub(r"\n---\n", "\n" + CORPUS_CONFIGS + "---\n", body, count=1)
    (CARDS / "corpus_README.md").write_text(body, encoding="utf-8")
    return body


def main():
    api = HfApi()
    corpus = corpus_card()
    api.upload_file(
        path_or_fileobj=corpus.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=CORPUS,
        repo_type="dataset",
        commit_message="card: declare the parquet config so the viewer picks up data/corpus.parquet",
    )
    results = (CARDS / "results_README.md").read_text(encoding="utf-8")
    api.upload_file(
        path_or_fileobj=results.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=RESULTS,
        repo_type="dataset",
        commit_message="card: yaml front matter + one parquet config per table; define each metric's source",
    )
    print(f"corpus card: {len(corpus)} chars, configs={'configs:' in corpus}")
    print(f"results card: {len(results)} chars, configs={'configs:' in results}")
    print(api.dataset_info(CORPUS).card_data)
    print(api.dataset_info(RESULTS).card_data)


if __name__ == "__main__":
    main()
