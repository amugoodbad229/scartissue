"""Render the research article to a static page inside the project's Hugging Face Space.

Run: uv run python -m experiments.build_space_article

The Space has no blog eligibility requirement, so this is the public written account until a real
Hugging Face blog URL exists. It is one page, built from `_research/HF-ARTICLE.md`, and it reuses the
Space stylesheet rather than introducing a second design.
"""

from __future__ import annotations

import html
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "_research" / "HF-ARTICLE.md"
TARGET = ROOT / "space" / "article.html"

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title} — ScarTissue</title>
<meta name="description" content="How weight-space repairs trade harmful compliance for over-refusal and lost task performance in Cohere Tiny Aya Global." />
<link rel="stylesheet" href="style.css" />
</head>
<body>
<a class="skip-link" href="#main">Skip to the article</a>
<header class="hero">
  <div class="topbar">
    <a class="brand" href="./" aria-label="ScarTissue home"><span class="brand-mark">S</span><span>ScarTissue</span></a>
    <nav aria-label="Site">
      <a href="./">Evidence gate</a>
      <a href="article.html" aria-current="page">Article</a>
    </nav>
  </div>
  <div class="hero-grid">
    <div>
      <p class="eyebrow">RESEARCH ARTICLE</p>
      <h1>{title}</h1>
      <p class="hero-lede">A recorded study of safety drift and weight-space repair, written for readers who
      need the trade-offs rather than a product claim.</p>
      <div class="hero-links"><a href="https://github.com/amugoodbad229/scartissue">Repository and reproduction</a><span>Exploratory evidence · no safety certification</span></div>
    </div>
    <aside class="hero-status" aria-label="Study status">
      <span class="status-dot" aria-hidden="true"></span>
      <p class="status-label">Status of this account</p>
      <strong>Exploratory</strong>
      <p>Judge calibration failed, the blind human audit and the locked held-out comparison are pending,
      and nothing here is a production safety result.</p>
    </aside>
  </div>
</header>
<main id="main" class="article-main" tabindex="-1">
  <article class="prose">
{body}
  </article>
  <p class="article-note">This page is a written account hosted on the project's Hugging Face Space. It is not a
  Hugging Face blog article.</p>
</main>
</body>
</html>
"""


def render() -> str:
    source = SOURCE.read_text(encoding="utf8")
    assert "SUBMISSION-FORM" not in source
    first = next(line for line in source.splitlines() if line.startswith("# "))
    title = first[2:].strip()
    # The hero already renders the title; keep it out of the body so it is not shown twice.
    body_md = source.replace(first, "", 1)
    body = markdown.markdown(body_md, extensions=["tables", "sane_lists"])
    assert "<table>" in body, "the results table must survive rendering"
    # A wide results table must scroll inside its own box instead of widening the page.
    body = body.replace("<table>", '<div class="table-scroll"><table>').replace("</table>", "</table></div>")
    return TEMPLATE.format(title=html.escape(title), body=body)


def main() -> None:
    page = render()
    TARGET.write_text(page, encoding="utf8")
    print(f"{TARGET.relative_to(ROOT)}: {len(page)} chars")


if __name__ == "__main__":
    main()
