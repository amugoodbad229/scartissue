"""Bounded, logged-out live deployment check; synthetic labels never become evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
URL = "https://scartissue-audit.vercel.app"
RESERVATION = "vercel_controls_live_cpu_20261110"
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("chromium")
         .uv_sync(str(ROOT), frozen=True, uv_version="0.12.23"))
app = modal.App("scartissue-vercel-smoke")


@app.function(image=image, cpu=1, memory=4096, timeout=300, retries=0, max_containers=1)
def smoke(expected_assets: dict, first: dict, sample_hash: str) -> dict:
    import re
    import shutil
    from playwright.sync_api import sync_playwright

    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "url": URL,
              "execution": "Modal CPU", "authenticated": False, "status": "running",
              "checks": [], "assets": {}, "screenshots": {}}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=shutil.which("chromium"), args=["--no-sandbox"])
            for width, height, name in ((1200, 900, "desktop"), (390, 844, "mobile")):
                context = browser.new_context(viewport={"width": width, "height": height}, accept_downloads=True)
                page = context.new_page()
                page.on("dialog", lambda dialog: dialog.accept())
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                response = page.goto(URL, wait_until="networkidle", timeout=60000)
                assert response.status == 200, f"Public HTTP status: {response.status}"
                page.get_by_role("heading", name="Can we trust the safety judge?").wait_for()
                assert page.url.startswith(URL), f"Unexpected redirect: {page.url}"
                assert "not pre-labelled" in page.locator("main").inner_text()
                result["screenshots"][f"vercel_audit_controls_intro_{name}.png"] = page.screenshot(full_page=True)
                for path, expected in expected_assets.items():
                    asset = context.request.get(URL + "/" + path)
                    assert asset.status == 200, f"Missing asset: {path}"
                    actual = hashlib.sha256(asset.body()).hexdigest()
                    assert actual == expected, f"Deployed asset differs from validated build: {path}"
                    result["assets"][path] = actual
                page.get_by_role("button", name="Start labelling", exact=True).click()
                assert page.locator("#prompt").inner_text() == first["prompt"]
                assert page.locator("#answer").inner_text() == re.sub(r"\*\*([^*]+)\*\*", r"\1", first["answer"])
                assert page.locator("#answer strong").all_text_contents() == re.findall(r"\*\*([^*]+)\*\*", first["answer"])
                panel = page.locator(".actionbar").bounding_box()
                assert panel["y"] + panel["height"] < page.locator("#prompt").bounding_box()["y"]
                assert panel["y"] + panel["height"] <= height
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                for button in page.locator(".choice").all():
                    box = button.bounding_box()
                    assert 0 <= box["y"] and box["y"] + box["height"] <= height
                    assert button.evaluate("el => { const r = el.getBoundingClientRect(); return el.contains(document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2)); }")
                result["screenshots"][f"vercel_audit_controls_scrolled_{name}.png"] = page.screenshot()
                page.get_by_role("button", name="Safe", exact=False).click()
                assert page.locator("h1").inner_text().startswith("Item 2 ")
                page.reload()
                page.get_by_role("button", name="Resume labelling").click()
                page.get_by_role("button", name="Progress & export").click()
                with page.expect_download() as download:
                    page.get_by_role("button", name="Download partial backup").click()
                data = json.loads(Path(download.value.path()).read_text())
                assert data == {"schema": "scartissue.human-audit/1", "sample_sha256": sample_hash,
                                "labels": [{"item_id": first["item_id"], "answer_sha256": first["answer_sha256"], "label": "safe"}]}
                assert not errors, errors
                result["checks"].append({"viewport": [width, height], "http_status": response.status,
                                          "public_without_login": True, "recorded_text_matches": True,
                                          "no_horizontal_overflow": True, "persistence_and_export": True,
                                          "bold_rendered": True, "label_controls_above_content": True,
                                          "label_controls_visible_and_clickable_after_scroll": True,
                                          "page_errors": errors})
                context.close()
            browser.close()
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


@app.local_entrypoint()
def main():
    budget = json.loads((ROOT / "results/modal_budget_20261105.json").read_text())
    reserve = [r for r in budget["reservations"] if r["job"] == RESERVATION]
    assert len(reserve) == 1 and reserve[0]["reserved_usd"] >= 1 and not reserve[0]["gpu"]
    assert sum(r["reserved_usd"] for r in budget["reservations"]) <= budget["total_ceiling"]
    sample = json.loads((ROOT / "human_audit/app/src/sample.json").read_text(encoding="utf8"))
    dist = ROOT / "human_audit/app/dist"
    assets = {p.relative_to(dist).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (dist / "assets").iterdir() if p.suffix in {".js", ".css"}}
    assert assets, "Build assets missing"
    result = smoke.remote(assets, sample["items"][0], sample["sample_sha256"])
    for name, content in result.pop("screenshots").items():
        (ROOT / "results" / name).write_bytes(content)
    path = ROOT / "results/vercel_controls_live_20261110.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(f"Live Vercel check failed; inspect {path}")
