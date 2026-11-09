"""Reader-facing Hub card guards. Source copy checks, not a scientific verdict."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "results" / "cards"
PUBLISHED = ("model_README.md", "results_README.md", "evals_README.md",
             "repair_patches_README.md", "abliteration_patches_README.md")


def test_cards_have_frontmatter_and_no_identifier_wall():
    for name in PUBLISHED:
        text = (CARDS / name).read_text(encoding="utf8")
        assert text.startswith("---\n") and "\n---\n" in text[4:], name
        assert not re.search(r"\b[0-9a-f]{40,}\b", text), f"{name} shows commit/blob hashes"
        assert "SUBMISSION-FORM" not in text, name


def test_cards_disclose_exploratory_evidence_and_real_loader_limits():
    model = (CARDS / "model_README.md").read_text(encoding="utf8").lower()
    for phrase in ("not a standalone model", "calibration failed", "exploratory",
                   "pending", "transactional rollback"):
        assert phrase in model, phrase
    results = (CARDS / "results_README.md").read_text(encoding="utf8")
    for config in ("safety_generations", "judge_verdicts", "capability", "patches", "frontier"):
        assert f"config_name: {config}" in results, config
        assert f"data/{config}.parquet" in results, config
