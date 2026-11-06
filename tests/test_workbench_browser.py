"""Real Chromium checks; the project's test execution policy is Modal-only."""
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
import threading

from playwright.sync_api import sync_playwright
import pytest

from experiments.evidence_bundle import ROOT, build_bundle


@pytest.fixture(scope="module")
def study_server():
    bundle = build_bundle()
    raw = json.dumps(bundle).encode()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/data/evidence.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(raw)
            else:
                super().do_GET()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(ROOT / "space-static")))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", bundle
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def page(study_server):
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=shutil.which("chromium"), args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(study_server[0])
        page.wait_for_function("document.getElementById('export').disabled === false")
        yield page
        browser.close()


def upload(page, value):
    page.locator("#bundle-file").set_input_files({"name": "study.json", "mimeType": "application/json", "buffer": json.dumps(value).encode()})


def test_real_study_abstains_export_replays_and_reflows(page, study_server):
    assert page.locator("#decisions .insufficient_evidence").count() == len([a for a in study_server[1]["arms"] if a["kind"] == "repair"])
    assert page.locator("#decisions .accept").count() == 0
    assert page.locator("#frontier-table tbody tr").count() == len(study_server[1]["arms"])
    assert page.locator("#replay details[open]").count() == 0
    page.locator("#replay > details > summary").click()
    assert page.locator("#replay .prompt").is_visible()
    page.locator("#replay article summary").first.click()
    assert page.locator("#replay .comp").first.is_visible()
    with page.expect_download() as info:
        page.locator("#export").click()
    packet = json.loads(Path(info.value.path()).read_text())
    assert packet["schema"] == "scartissue.decision/1"
    assert len(packet["bundle_sha256"]) == 64
    assert all("completions" not in a for a in packet["metrics"])
    assert all(d["status"] != "accept" for d in packet["decisions"])
    with page.expect_download() as summary_info:
        page.locator("#export-summary").click()
    summary = Path(summary_info.value.path()).read_text()
    assert "insufficient_evidence" in summary and packet["bundle_sha256"] in summary
    assert all(s["sha256"] in summary for s in packet["sources"])
    assert study_server[1]["arms"][0]["completions"][0]["answer"] not in summary
    page.locator("#max-harm").fill("0.01")
    assert page.locator("#export").is_disabled()
    assert page.locator("#export-summary").is_disabled()
    page.get_by_role("button", name="Evaluate repairs").click()
    assert page.locator("#export").is_enabled()
    page.locator("details[open]").evaluate_all("xs => xs.forEach(e => e.open = false)")
    page.screenshot(path="/tmp/scartissue_workbench_desktop.png", full_page=True)
    page.set_viewport_size({"width": 320, "height": 740})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Horizontal page overflow at narrow width"
    page.screenshot(path="/tmp/scartissue_workbench_mobile.png", full_page=True)


def test_import_rejects_tampering_preserves_scope_and_escapes_labels(page, study_server):
    original_study = page.locator("#study").inner_text()
    data = json.loads(json.dumps(study_server[1]))
    data["prompts"][0]["text"] += " altered"
    upload(page, data)
    page.wait_for_function("document.getElementById('status').textContent.includes('hash mismatch')")
    assert page.locator("#study").inner_text() == original_study
    data = json.loads(json.dumps(study_server[1]))
    data["arms"][0]["id"] = '<img src=x onerror="window.__xss=1">'
    upload(page, data)
    page.wait_for_function("document.getElementById('status').textContent.startsWith('Policy evaluated')")
    assert page.locator("#frontier-table img").count() == 0
    assert page.evaluate("window.__xss") is None
    page.locator("#expected-revision").fill("different-revision")
    page.get_by_role("button", name="Evaluate repairs").click()
    assert "differs" in page.locator("#decisions").inner_text()
    assert page.locator("#decisions .accept").count() == 0


def test_guided_walkthrough_links_and_policy_outcome(page):
    guide = page.get_by_role("navigation", name="Judge walkthrough")
    assert "sparse and global repair" in guide.inner_text()
    assert "Tighten maximum harm-ASR" in guide.inner_text()
    assert "judge disagreement" in guide.inner_text()
    assert guide.locator('a[href^="#"]').evaluate_all(
        "xs => xs.every(a => document.getElementById(a.hash.slice(1)))")
    page.get_by_label("Maximum harm-ASR (0–1)").fill("0.01")
    page.get_by_role("button", name="Evaluate repairs").click()
    assert page.locator("#decisions .accept").count() == 0
    assert "insufficient" in page.locator("#decisions").inner_text().lower()
    assert page.get_by_role("button", name="Download readable summary").is_enabled()


def test_keyboard_skip_labels_and_text_resize(page):
    page.keyboard.press("Control+Home")
    page.locator("body").click(position={"x": 1, "y": 1})
    page.keyboard.press("Tab")
    assert page.locator(".skip-link").evaluate("e => e === document.activeElement")
    page.keyboard.press("Enter")
    assert page.locator("#main").evaluate("e => e === document.activeElement")
    page.get_by_label("Maximum harm-ASR (0–1)").focus()
    page.keyboard.press("Control+A")
    page.keyboard.type("0.02")
    assert page.locator("#export").is_disabled()
    page.get_by_role("button", name="Evaluate repairs").focus()
    page.keyboard.press("Enter")
    assert page.locator("#export").is_enabled()
    assert page.locator("input:not([type=file]), select").evaluate_all("xs => xs.every(e => e.labels.length > 0)")
    page.add_style_tag(content="body { font-size: 200%; }")
    page.set_viewport_size({"width": 320, "height": 740})
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "Overflow with enlarged text"


def test_computed_contrast_and_focus(page):
    def contrast(first, second):
        def luminance(color):
            rgb = [int(v) / 255 for v in re.findall(r"\d+", color)[:3]]
            linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
            return sum(v * w for v, w in zip(linear, (0.2126, 0.7152, 0.0722)))
        a, b = sorted([luminance(first), luminance(second)])
        return (b + 0.05) / (a + 0.05)

    checks = []
    for selector, background in [("body", "body"), (".note", "body"),
                                 (".warning", ".warning"), (".tag", ".tag"),
                                 ("#max-harm", "#max-harm"), ("#preset", "#preset"),
                                 ("button[type=submit]", "button[type=submit]")]:
        foreground = page.locator(selector).first.evaluate("e => getComputedStyle(e).color")
        surface = page.locator(background).first.evaluate("e => getComputedStyle(e).backgroundColor")
        ratio = contrast(foreground, surface)
        checks.append({"selector": selector, "kind": "text", "ratio": ratio, "minimum": 4.5})
        assert ratio >= 4.5, (selector, ratio)
    for selector in ("#max-harm", "#preset", "#export"):
        border = page.locator(selector).evaluate("e => getComputedStyle(e).borderTopColor")
        surface = page.locator(selector).evaluate("e => getComputedStyle(e).backgroundColor")
        ratio = contrast(border, surface)
        checks.append({"selector": selector, "kind": "control boundary", "ratio": ratio, "minimum": 3})
        assert ratio >= 3, (selector, ratio)
    page.get_by_label("Maximum harm-ASR (0–1)").focus()
    page.keyboard.press("Tab")
    focus = page.evaluate("getComputedStyle(document.activeElement).outlineColor")
    ratio = contrast(focus, "rgb(255, 255, 255)")
    assert ratio >= 3
    checks.append({"kind": "keyboard focus", "ratio": ratio, "minimum": 3})
    Path("/tmp/scartissue_workbench_accessibility.json").write_text(json.dumps({
        "execution": "Modal Chromium computed-style checks", "contrast_checks": checks,
        "scope": "Selected text, control boundaries and focus; keyboard, reflow and errors tested separately",
        "not_checked": ["Screen-reader traversal", "Every browser/OS", "Full WCAG conformance"]}, indent=2))


def test_invalid_bundle_and_fetch_error_are_not_safe(page, study_server):
    upload(page, {"schema": "not-supported"})
    page.wait_for_function("document.getElementById('status').textContent.includes('Unsupported evidence schema')")
    assert page.locator("#decisions .accept").count() == 0
    page.route("**/data/evidence.json", lambda route: route.fulfill(status=503, body="Unavailable"))
    page.reload()
    page.wait_for_function("document.getElementById('status').textContent.includes('503')")
    assert page.locator("#export").is_disabled()
    assert page.locator("#preset").is_disabled()
