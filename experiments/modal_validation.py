"""Bounded CPU validation on Modal; no model downloads, secrets, or Volume writes.

uv run python -m modal run experiments/modal_validation.py --reservation-id <pre-reserved-job>
Name the log and reserve the run in results/modal_budget_20261105.json before launch.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "ScarTissue/repos/AuditKIT"
AUDITKIT_COMMIT = "71316da345784ff220d5ed59337441a59ab31526"
REMOTE = "/work/scaartissue"
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("nodejs", "chromium")
         .uv_sync(str(ROOT), frozen=True, uv_version="0.12.23")
         .add_local_dir(str(ROOT / "experiments"), f"{REMOTE}/experiments",
                        ignore=["__pycache__", "*.pyc"])
         .add_local_dir(str(ROOT / "tests"), f"{REMOTE}/tests", ignore=["__pycache__", "*.pyc"])
         .add_local_dir(str(ROOT / "artifacts"), f"{REMOTE}/artifacts",
                        ignore=["__pycache__", "*.pyc"])
         .add_local_dir(str(ROOT / "results"), f"{REMOTE}/results")
         .add_local_dir(str(ROOT / "1-curate"), f"{REMOTE}/1-curate")
         .add_local_dir(str(ROOT / "space"), f"{REMOTE}/space")
         .add_local_dir(str(ROOT / "_research"), f"{REMOTE}/_research")
         .add_local_dir(str(ROOT / "report"), f"{REMOTE}/report",
                        ignore=["*.pdf", "*.png", "__pycache__", "*.pyc"])
         .add_local_file(str(ROOT / "README.md"), f"{REMOTE}/README.md")
         .add_local_file(str(ROOT / "REPRODUCE.md"), f"{REMOTE}/REPRODUCE.md")
         .add_local_dir(str(ROOT / "human_audit"), f"{REMOTE}/human_audit", ignore=["__pycache__"])
         .add_local_dir(str(UPSTREAM / "src"), "/work/ScarTissue/repos/AuditKIT/src",
                        ignore=["__pycache__", "*.pyc"])
         .add_local_file(str(UPSTREAM / "LICENSE.md"), "/work/ScarTissue/repos/AuditKIT/LICENSE.md")
         .add_local_file(str(ROOT / "pyproject.toml"), f"{REMOTE}/pyproject.toml")
         .add_local_file(str(ROOT / "uv.lock"), f"{REMOTE}/uv.lock"))
app = modal.App("scartissue-validation")


def source_hashes(root: Path) -> dict[str, str]:
    paths = [root / "pyproject.toml", root / "uv.lock"]
    for folder in ("experiments", "tests", "artifacts", "space", "_research", "report", "human_audit"):
        paths.extend(p for p in (root / folder).rglob("*")
                     if p.suffix in {".py", ".js", ".cjs", ".html", ".css", ".md", ".tex"}
                     and "__pycache__" not in p.parts)
    paths.extend(p for name in ("README.md", "REPRODUCE.md")
                 if (p := root / name).exists())
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def verify_snapshot(root: Path, expected: dict[str, str]) -> None:
    actual = source_hashes(root)
    if actual != expected:
        changed = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
        raise ValueError(f"Source snapshot mismatch: {changed}")


@app.function(image=image, cpu=1, memory=4096, timeout=600, retries=0, max_containers=1)
def validate(expected: dict[str, str], auditkit_hashes: dict[str, str]) -> dict:
    import importlib.metadata
    import os
    import platform

    remote_root = Path(REMOTE)
    verify_snapshot(remote_root, expected)
    upstream = remote_root.parent / "ScarTissue/repos/AuditKIT"
    actual = {p.relative_to(upstream).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((upstream / "src").rglob("*.py"))}
    actual["LICENSE.md"] = hashlib.sha256((upstream / "LICENSE.md").read_bytes()).hexdigest()
    if actual != auditkit_hashes:
        raise ValueError("AuditKIT source/license snapshot mismatch")
    started = time.monotonic()
    output = {"status": "running", "execution": "Modal CPU", "source_sha256": expected,
              "auditkit_commit": AUDITKIT_COMMIT, "auditkit_source_sha256": actual,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": {k: importlib.metadata.version(k) for k in ("pytest", "numpy", "modal")}},
              "resources": {"cpu": 1, "memory_mib": 4096, "timeout_seconds": 600, "gpu": None},
              "runs": []}
    # Generated artifacts must exist before tests check documentation links to them.
    sys.path.insert(0, str(remote_root))
    from experiments.evidence_bundle import build_bundle
    from experiments.verify_submission import verify, build_readiness
    output["evidence_bundle"] = build_bundle(remote_root)
    output["submission_verification"] = verify(remote_root, output["evidence_bundle"])
    output["readiness_ledger"] = build_readiness(remote_root, output["evidence_bundle"], output["submission_verification"])
    (remote_root / "results/readiness_ledger.json").write_text(
        json.dumps(output["readiness_ledger"], indent=2, allow_nan=False) + "\n", encoding="utf8")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for command in (["pytest", "-q"], [sys.executable, "-m", "pytest", "-q"]):
        tick = time.monotonic()
        try:
            result = subprocess.run(command, cwd=REMOTE, env=env, capture_output=True, text=True, timeout=240)
            run = {"command": command, "exit_code": result.returncode,
                   "stdout": result.stdout, "stderr": result.stderr}
        except subprocess.TimeoutExpired as exc:
            run = {"command": command, "exit_code": None, "error": "test invocation timed out",
                   "stdout": (exc.stdout or b"").decode(errors="replace"),
                   "stderr": (exc.stderr or b"").decode(errors="replace")}
        run["elapsed_seconds"] = time.monotonic() - tick
        output["runs"].append(run)
    output["elapsed_seconds"] = time.monotonic() - started
    output["environment"]["node"] = subprocess.check_output(["node", "--version"], text=True).strip()
    output["status"] = "passed" if all(r["exit_code"] == 0 for r in output["runs"]) else "failed"
    if output["submission_verification"]["status"] != "passed":
        output["status"] = "failed"
    output["browser_screenshots"] = {p.name: p.read_bytes() for p in Path("/tmp").glob("scartissue_workbench_*.png")}
    audit = Path("/tmp/scartissue_workbench_accessibility.json")
    if audit.exists():
        output["browser_accessibility"] = json.loads(audit.read_text())
    return output


@app.local_entrypoint()
def main(reservation_id: str):
    head = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain"], text=True).strip()
    if head != AUDITKIT_COMMIT or dirty:
        raise ValueError("Need the clean pinned AuditKIT checkout; never substitute mocks")
    budget_path = ROOT / "results/modal_budget_20261105.json"
    budget = json.loads(budget_path.read_text(encoding="utf8"))
    if budget["status"] != "approved":
        raise ValueError("Modal spending is not authorized")
    reservations = [r for r in budget["reservations"] if r["job"] == reservation_id]
    if len(reservations) != 1 or reservations[0]["reserved_usd"] < 1 or reservations[0]["gpu"]:
        raise ValueError("Pre-reserve this CPU job before invoking Modal (including image build)")
    if sum(r["reserved_usd"] for r in budget["reservations"]) > budget["total_ceiling"]:
        raise ValueError("Modal authorization reserve exhausted")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / f"results/modal_validation_{stamp}.json"
    hashes = {p.relative_to(UPSTREAM).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((UPSTREAM / "src").rglob("*.py"))}
    hashes["LICENSE.md"] = hashlib.sha256((UPSTREAM / "LICENSE.md").read_bytes()).hexdigest()
    result = {"status": "running", "created_at": stamp, "reservation_id": reservation_id,
              "modal_profile": "scartissue-replacement", "source_sha256": source_hashes(ROOT)}
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    try:
        call = validate.spawn(result["source_sha256"], hashes)
        result["function_call_id"] = call.object_id
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
        result.update(call.get())
        screenshots = result.pop("browser_screenshots", {})
        for name, raw in screenshots.items():
            (ROOT / "results" / name).write_bytes(raw)
        result["browser_screenshot_paths"] = [f"results/{name}" for name in screenshots]
        accessibility = result.pop("browser_accessibility", None)
        if accessibility is not None:
            audit_path = ROOT / "results/workbench_accessibility_20261106.json"
            audit_path.write_text(json.dumps(accessibility, indent=2) + "\n", encoding="utf8")
            result["browser_accessibility_path"] = audit_path.relative_to(ROOT).as_posix()
        verification = result.pop("submission_verification", None)
        if verification is not None:
            verification_path = ROOT / f"results/submission_verification_{stamp}.json"
            verification_content = json.dumps(verification, indent=2, allow_nan=False) + "\n"
            verification_path.write_text(verification_content, encoding="utf8")
            (ROOT / "results/submission_verification.json").write_text(verification_content, encoding="utf8")
            result["submission_verification_path"] = verification_path.relative_to(ROOT).as_posix()
        ledger = result.pop("readiness_ledger", None)
        if ledger is not None:
            content = json.dumps(ledger, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            ledger_path = ROOT / f"results/readiness_ledger_{stamp}.json"
            ledger_path.write_text(content, encoding="utf8")
            (ROOT / "results/readiness_ledger.json").write_text(content, encoding="utf8")
            static_ledger = ROOT / "space/data/readiness.json"
            static_ledger.parent.mkdir(parents=True, exist_ok=True)
            static_ledger.write_text(content, encoding="utf8")
            result["readiness_ledger_path"] = ledger_path.relative_to(ROOT).as_posix()
        bundle = result.pop("evidence_bundle", None)
        if result["status"] == "passed" and bundle is not None:
            content = json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
            bundle_path = ROOT / "results/workbench_evidence_20261106.json"
            bundle_path.write_text(content, encoding="utf8")
            static_path = ROOT / "space/data/evidence.json"
            static_path.parent.mkdir(parents=True, exist_ok=True)
            static_path.write_text(content, encoding="utf8")
            result["evidence_bundle_path"] = bundle_path.relative_to(ROOT).as_posix()
            result["evidence_bundle_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(f"{path}: {result['status']}", flush=True)
    if result["status"] != "passed":
        raise RuntimeError(f"Modal validation failed; inspect {path}")
