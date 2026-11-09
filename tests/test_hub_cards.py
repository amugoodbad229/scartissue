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


def test_space_article_page_is_generated_from_the_article_and_linked():
    import re
    page = (ROOT / "space" / "article.html").read_text(encoding="utf8")
    article = (ROOT / "_research" / "HF-ARTICLE.md").read_text(encoding="utf8")
    body = page.split('<article class="prose">', 1)[1]
    for heading in ("The research question", "What we compared", "What the results show"):
        assert heading in body, heading
    assert "<table>" in body and "Judged harm rate" in body
    # One rendered h2 per markdown section: the page is derived, not a second copy.
    assert article.count("\n## ") == body.count("<h2>")
    assert "not a Hugging Face blog article" in re.sub(r"\s+", " ", page)
    assert "article.html" in (ROOT / "space" / "index.html").read_text(encoding="utf8")
    assert "article.html" in (ROOT / "README.md").read_text(encoding="utf8")
