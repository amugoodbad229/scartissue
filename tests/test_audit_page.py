"""The blind audit page must actually work: label, navigate, persist, export verifiable labels."""
import json
from pathlib import Path
import shutil

from playwright.sync_api import sync_playwright
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "human_audit/audit.html"


@pytest.fixture(scope="module")
def sample() -> dict:
    return json.loads((ROOT / "results/human_audit_sample_20261106.json").read_text(encoding="utf8"))


@pytest.fixture
def page(tmp_path):
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=shutil.which("chromium"), args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1100, "height": 900}, accept_downloads=True)
        page = context.new_page()
        page.goto(PAGE.as_uri())
        page.wait_for_function("document.getElementById('prompt') !== null && document.getElementById('prompt').textContent.length > 0")
        yield page
        browser.close()


def test_page_shows_blind_items_and_counts(page, sample):
    assert page.locator("#prompt").inner_text().strip()
    assert page.locator("#answer").inner_text().strip()
    assert str(len(sample["items"])) in page.locator("#meta").inner_text()
    assert page.locator("#progress").inner_text().startswith("1 / " + str(len(sample["items"])))
    body = page.content()
    for leak in ("eval_malicious", "eval_benign", "harm2_a4", "global_a4", '"arm"'):
        assert leak not in body


def test_keyboard_labels_persist_and_export(page, sample, tmp_path):
    total = len(sample["items"])
    page.keyboard.press("1")
    assert page.locator("#progress").inner_text().startswith("2 /")
    page.keyboard.press("ArrowLeft")
    assert page.locator("#b-harmful").get_attribute("aria-pressed") == "true"
    for _ in range(total):
        page.keyboard.press("2")
    assert page.locator("#mark").inner_text() == "all labelled"
    assert page.locator("#export").is_enabled()
    with page.expect_download() as info:
        page.click("#export")
    path = tmp_path / "labels.json"
    info.value.save_as(path)
    payload = json.loads(path.read_text(encoding="utf8"))
    assert payload["schema"] == "scartissue.human-audit/1"
    assert payload["sample_sha256"] == sample["sample_sha256"]
    assert len(payload["labels"]) == total
    known = {item["item_id"]: item["answer_sha256"] for item in sample["items"]}
    assert {row["item_id"] for row in payload["labels"]} == set(known)
    for row in payload["labels"]:
        assert row["answer_sha256"] == known[row["item_id"]]
        assert row["label"] in ("harmful", "safe", "unsure")


def test_labels_survive_reload(page, sample):
    first = page.locator("#prompt").inner_text()
    page.keyboard.press("3")
    page.reload()
    page.wait_for_function("document.getElementById('prompt').textContent.length > 0")
    assert page.locator("#progress").inner_text().startswith("2 /")
    page.keyboard.press("ArrowLeft")
    assert page.locator("#prompt").inner_text() == first
    assert page.locator("#b-unsure").get_attribute("aria-pressed") == "true"


def test_export_is_disabled_until_every_item_is_labelled(page, sample):
    assert page.locator("#export").is_disabled()
    page.keyboard.press("2")
    assert page.locator("#export").is_disabled()
    assert page.locator("#mark").inner_text() == "1 / " + str(len(sample["items"])) + " labelled"


def test_exported_labels_flow_into_analysis(page, sample, tmp_path):
    """The page's own export must be accepted by the analyser (mechanics, not a real audit)."""
    from experiments import analyse_audit

    total = len(sample["items"])
    for i in range(total):
        page.keyboard.press("1" if i % 2 else "2")
    assert page.locator("#mark").inner_text() == "all labelled"
    with page.expect_download() as info:
        page.click("#export")
    path = tmp_path / "labels.json"
    info.value.save_as(path)
    result = analyse_audit.main(str(path), str(tmp_path / "analysis.json"))
    assert "gate_passed" in result
    stored = json.loads((tmp_path / "analysis.json").read_text(encoding="utf8"))
    assert stored["labels"]["path"] == str(path)
    assert stored["sample"]["usable"] == total
