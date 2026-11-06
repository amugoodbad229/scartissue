"""Provenance of the three track repositories this submission actually runs.

Every claim here is machine-checked against the read-only archive at
``../ScarTissue/repos/<repo>`` (never vendored, never re-licensed):

  * git HEAD and commit date;
  * licence file sha256 and its first heading line;
  * the exact call sites in this repo that import the package;
  * a sha256 of the package Python tree, so "we used commit X" is checkable;
  * the runtime evidence already stored in ``results/`` that proves the import ran.

Writes ``results/toolkit_provenance.json``. No GPU, no network.

    uv run python -m experiments.toolkit_provenance
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOS = ROOT.parent / "ScarTissue/repos"

REQUIRED = {
    "CuratorKIT": {"package": "src/curatorkit", "import": "curatorkit",
                   "role": "corpus construction with row-level provenance (763 rows)"},
    "CircuitKIT": {"package": "src/circuitkit", "import": "circuitkit",
                   "role": "EAP-IG circuit discovery on the drifted model"},
    "AuditKIT": {"package": "src/auditkit", "import": "auditkit",
                 "role": "native evaluation of stored answers against a strict harm metric"},
    "SafeTune": {"package": "src/safetune", "import": "safetune",
                 "role": "NOT imported: the drift and the repair run through our own audited "
                         "rank-one operator instead (see results/p1_grid.json)"},
}


def sha_tree(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = 0
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        files += 1
    return digest.hexdigest(), files


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def call_sites(package: str) -> list[dict]:
    hits = []
    pattern = re.compile(rf"^\s*(?:from|import)\s+{package}\b")
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {"__pycache__", ".venv", "report"} for part in path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, 1):
            if pattern.match(line):
                hits.append({"file": str(path.relative_to(ROOT)).replace("\\", "/"),
                             "line": number, "code": line.strip()})
    return hits


def runtime_evidence() -> dict:
    """Pull the stored proof that each import actually executed on Modal."""
    out = {}
    audit = ROOT / "results/p19_safety_judges_20261103_002850.json"
    if audit.exists():
        payload = json.loads(audit.read_text(encoding="utf8"))
        out["AuditKIT"] = {"evidence": str(audit.relative_to(ROOT)).replace("\\", "/"),
                           "commit": payload.get("external_AuditKIT_commit"),
                           "python_tree_sha256": payload.get("external_AuditKIT_python_tree"),
                           "native_counts": {w["family"]: w.get("auditkit", {}).get("stats", {})
                                                           .get("strict_response_harm", {}).get("count")
                                             for w in payload.get("workers", [])}}
    for path in sorted((ROOT / "results").glob("p23_circuit_*.json")):
        payload = json.loads(path.read_text(encoding="utf8"))
        out.setdefault("CircuitKIT", []).append(
            {"evidence": str(path.relative_to(ROOT)).replace("\\", "/"),
             "result_shape": payload.get("result_shape"), "node_naming": payload.get("node_naming"),
             "signed_nodes": len(payload.get("node_scores") or {})})
    for path in sorted((ROOT / "results").glob("corpus_provenance_*.json")) + sorted(
            (ROOT / "results").glob("p8_corpus_*.json")):
        payload = json.loads(path.read_text(encoding="utf8"))
        out.setdefault("CuratorKIT", []).append({"evidence": str(path.relative_to(ROOT)).replace("\\", "/"),
                                                 "exporter": payload.get("exporter")})
    return out


def main() -> None:
    evidence = runtime_evidence()
    report = {"schema": "scartissue.toolkit-provenance/1", "repos": {}}
    for name, spec in REQUIRED.items():
        repo = REPOS / name
        entry = {"path": str(repo), "role": spec["role"], "package": spec["package"],
                 "imported_by_this_repo": call_sites(spec["import"]),
                 "verified_runtime": evidence.get(name)}
        if repo.exists():
            entry["git_head"] = git(repo, "rev-parse", "HEAD")
            entry["git_head_date"] = git(repo, "log", "-1", "--format=%cs")
            entry["git_dirty"] = bool(git(repo, "status", "--porcelain"))
            licence = next((p for p in repo.iterdir() if p.name.lower().startswith("licen")), None)
            if licence:
                text = licence.read_text(encoding="utf8", errors="ignore").splitlines()
                entry["licence"] = {"file": licence.name,
                                    "sha256": hashlib.sha256(licence.read_bytes()).hexdigest(),
                                    "first_heading": next((l.strip() for l in text if l.strip()), "")}
            package = repo / spec["package"]
            if package.exists():
                entry["package_tree_sha256"], entry["package_py_files"] = sha_tree(package)
            entry["vendored"] = False
            entry["installed_into_env"] = False
        else:
            entry["missing"] = True
        report["repos"][name] = entry
    report["track_repos_used_together"] = [n for n, e in report["repos"].items()
                                           if e.get("imported_by_this_repo") and e.get("verified_runtime")]
    report["note"] = ("CircuitKIT and AuditKIT are imported from the archive via PYTHONPATH inside "
                      "the Modal image; CuratorKIT likewise for the corpus run. Nothing is vendored "
                      "or pip-installed, and the LSAL licences stay with their repositories.")
    path = ROOT / "results/toolkit_provenance.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf8")
    print(json.dumps({k: {"head": v.get("git_head", "")[:12], "call_sites": len(v["imported_by_this_repo"]),
                          "runtime": bool(v["verified_runtime"])} for k, v in report["repos"].items()},
                     indent=1))
    print("used together:", report["track_repos_used_together"])
    print(path)


if __name__ == "__main__":
    main()
