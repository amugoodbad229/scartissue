"""Production React audit mechanics. Synthetic labels are tests, never human evidence."""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import threading

from playwright.sync_api import sync_playwright
import pytest

from experiments.analyse_audit import load_labels, main as analyse

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "human_audit/app"
SAMPLE = json.loads((ROOT / "results/human_audit_sample_20261106.json").read_text(encoding="utf8"))
KEY = "scartissue-audit-" + SAMPLE["sample_sha256"]


@pytest.fixture(scope="module")
def url():
    assert (APP / "dist/index.html").exists(), "Build the React audit before Modal validation"
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(APP / "dist"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def page(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=shutil.which("chromium"), args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1200, "height": 900}, accept_downloads=True)
        page.on("dialog", lambda dialog: dialog.accept())
        page.goto(url)
        page.get_by_role("heading", name="Can we trust the safety judge?").wait_for()
        yield page
        browser.close()


def start(page):
    page.get_by_role("button", name="Start labelling", exact=True).click()
    page.locator("#prompt").wait_for()


def review(page):
    page.get_by_role("button", name="Progress & export").click()


def payload(count=None):
    items = SAMPLE["items"] if count is None else SAMPLE["items"][:count]
    return {"schema": "scartissue.human-audit/1", "sample_sha256": SAMPLE["sample_sha256"],
            "labels": [{"item_id": item["item_id"], "answer_sha256": item["answer_sha256"],
                        "label": "harmful" if i % 2 else "safe"} for i, item in enumerate(items)]}


def upload(page, data):
    page.locator("#backup").set_input_files({"name": "labels.json", "mimeType": "application/json",
                                           "buffer": json.dumps(data).encode()})


def test_blind_payload_matches_frozen_sample():
    blind = json.loads((APP / "src/sample.json").read_text(encoding="utf8"))
    gate = json.loads((ROOT / "results/calibration_gate_20261106.json").read_text(encoding="utf8"))
    fields = ("item_id", "prompt", "answer", "answer_sha256")
    assert set(blind) == {"sample_sha256", "items", "gate"}
    assert blind["sample_sha256"] == SAMPLE["sample_sha256"] == gate["sample_sha256"]
    assert blind["items"] == [{k: item[k] for k in fields} for item in SAMPLE["items"]]
    assert blind["gate"] == {k: gate[k] for k in ("minimum_labelled_items", "minimum_class_items", "thresholds")}
    bundle = "".join(p.read_text(encoding="utf8") for p in (APP / "dist/assets").glob("*.js"))
    # React DOM includes case"role":; reject dataset metadata, not accessibility support.
    for leak in ('"arm":', "eval_malicious", "eval_benign", "harm2_a4", "global_a4"):
        assert leak not in bundle
    for item in SAMPLE["items"]:
        assert item["answer_sha256"] in bundle


def test_intro_explains_recorded_answers_and_no_automatic_analysis(page):
    assert page.locator("#answer").count() == 0
    assert "not pre-labelled" in page.locator("main").inner_text()
    assert "Finishing the audit alone proves neither safety nor repair success" in page.locator("main").inner_text()
    assert page.get_by_role("link", name="Open the evidence workbench").count() == 1
    start(page)
    assert page.locator("#prompt").inner_text() == SAMPLE["items"][0]["prompt"]
    assert page.locator("#answer").inner_text() == re.sub(r"\*\*([^*]+)\*\*", r"\1", SAMPLE["items"][0]["answer"])
    assert page.locator("h1").evaluate("el => document.activeElement === el")


@pytest.mark.parametrize("width", [1200, 390, 320])
def test_label_controls_above_content_and_sticky(page, width):
    page.set_viewport_size({"width": width, "height": 900})
    start(page)
    panel = page.locator(".actionbar")
    bounds = panel.bounding_box()
    assert bounds["y"] + bounds["height"] < page.locator("#prompt").bounding_box()["y"]
    assert bounds["y"] + bounds["height"] <= 900
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    assert page.evaluate("window.scrollY") > 0
    for button in page.locator(".choice").all():
        box = button.bounding_box()
        assert 0 <= box["y"] and box["y"] + box["height"] <= 900
        assert button.evaluate("el => { const r = el.getBoundingClientRect(); return el.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2)); }")
    page.get_by_role("button", name="Safe", exact=False).click()
    assert page.locator("h1").inner_text().startswith("Item 2 ")
    assert page.evaluate("window.scrollY") == 0


def test_recorded_bold_is_rendered_without_changing_export(page, tmp_path):
    start(page)
    expected = re.findall(r"\*\*([^*]+)\*\*", SAMPLE["items"][0]["answer"])
    assert "Listen Actively" in expected
    assert page.locator("#answer strong").all_text_contents() == expected
    assert "**Listen Actively**" not in page.locator("#answer").inner_text()
    assert page.locator("#answer strong").first.evaluate("el => Number(getComputedStyle(el).fontWeight)") >= 700
    assert page.locator("#answer").evaluate("el => [...el.querySelectorAll('*')].every(child => child.tagName === 'STRONG')")
    page.keyboard.press("2")
    review(page)
    with page.expect_download() as download:
        page.get_by_role("button", name="Download partial backup").click()
    dest = tmp_path / "formatted-label.json"
    download.value.save_as(dest)
    assert json.loads(dest.read_text()) == payload(1)


def test_label_navigation_undo_and_reload(page):
    start(page)
    page.keyboard.press("1")
    assert page.locator("h1").inner_text().startswith("Item 2 ")
    page.keyboard.press("ArrowLeft")
    assert page.get_by_role("button", name="Harmful").get_attribute("aria-pressed") == "true"
    page.keyboard.press("z")
    assert "Unlabelled" in page.locator(".chip").inner_text()
    page.keyboard.press("3")
    page.reload()
    page.get_by_role("button", name="Resume labelling").click()
    assert page.locator("h1").inner_text().startswith("Item 2 ")
    page.keyboard.press("ArrowLeft")
    assert page.get_by_role("button", name="Unsure").get_attribute("aria-pressed") == "true"


def test_partial_backup_roundtrip_and_invalid_import_preserves_labels(page, tmp_path):
    start(page)
    page.keyboard.press("2")
    review(page)
    assert page.get_by_role("button", name="Download labels for analysis").count() == 0
    with page.expect_download() as download:
        page.get_by_role("button", name="Download partial backup").click()
    assert download.value.suggested_filename == "human_audit_partial_backup.json"
    dest = tmp_path / "partial.json"
    download.value.save_as(dest)
    data = json.loads(dest.read_text())
    assert data == payload(1)
    with pytest.raises(ValueError, match="unlabelled"):
        load_labels(dest, SAMPLE)
    page.get_by_role("button", name="Clear saved labels").click()
    review(page)
    upload(page, data)
    assert page.locator("h1").inner_text().startswith("Item 2 ")
    review(page)
    invalid = payload(1)
    invalid["labels"][0]["answer_sha256"] = "tampered"
    upload(page, invalid)
    assert "Could not restore" in page.get_by_role("alert").inner_text()
    assert page.locator(".stat").first.locator("strong").inner_text() == "0"
    assert page.locator(".stat").nth(1).locator("strong").inner_text() == "1"


@pytest.mark.parametrize("kind", ["sample", "duplicate", "unknown", "bad_label", "html"])
def test_import_rejects_bad_files(page, kind):
    review(page)
    data = payload(1)
    if kind == "sample":
        data["sample_sha256"] = "wrong"
    elif kind == "duplicate":
        data["labels"].append(dict(data["labels"][0]))
    elif kind == "unknown":
        data["labels"][0]["item_id"] = "unknown"
    elif kind == "bad_label":
        data["labels"][0]["label"] = "approved"
    else:
        data = "<img src=x onerror=alert(1)>"
    upload(page, data)
    assert "Could not restore" in page.get_by_role("alert").inner_text()
    assert page.get_by_role("button", name="Download partial backup").is_disabled()


def test_last_label_completion_export_and_analysis(page, tmp_path):
    review(page)
    upload(page, payload(len(SAMPLE["items"]) - 1))
    page.locator("#answer").wait_for()
    page.keyboard.press("1")
    page.get_by_role("heading", name="Labels complete. Analysis comes next.").wait_for()
    with page.expect_download() as download:
        page.get_by_role("button", name="Download labels for analysis").click()
    assert download.value.suggested_filename == "human_audit_labels.json"
    dest = tmp_path / "synthetic_labels.json"
    download.value.save_as(dest)
    assert len(load_labels(dest, SAMPLE)) == len(SAMPLE["items"])
    result = analyse(str(dest), str(tmp_path / "synthetic_analysis.json"))
    assert "gate_passed" in result  # No assertion of a scientific pass from fake labels.
    assert result["usable"] == len(SAMPLE["items"])
    page.get_by_role("button", name="Undo last label").click()
    assert "Unlabelled" in page.locator(".chip").inner_text()


def test_storage_failure_warns_and_still_exports(page):
    page.add_init_script("Storage.prototype.setItem = () => { throw new Error('blocked'); };")
    page.reload()
    page.get_by_role("alert").wait_for()
    start(page)
    page.keyboard.press("2")
    review(page)
    assert page.get_by_role("button", name="Download partial backup").is_enabled()


def test_mobile_reflow_and_screenshots(page):
    page.screenshot(path="/tmp/scartissue_audit_desktop.png", full_page=True)
    page.set_viewport_size({"width": 320, "height": 780})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.screenshot(path="/tmp/scartissue_audit_mobile.png", full_page=True)
    start(page)
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    for button in page.locator(".choice").all():
        assert button.bounding_box()["height"] >= 44
    page.screenshot(path="/tmp/scartissue_audit_item_mobile.png", full_page=True)
    page.add_style_tag(content="html { font-size: 200%; }")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
