"""Offline artifact integrity, not a certificate of scientific or submission readiness.

Execute on Modal via modal_validation.py; the local entrypoint saves returned results.
No HTTP availability checks, expected performance values or fixed report hashes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.evidence_bundle import ROOT, INPUTS, build_bundle


def artifact_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not relative or Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes repository: {relative}")
    return path


def comparable(bundle: dict) -> dict:
    return {key: value for key, value in bundle.items() if key != "created_at"}


GATE = "results/calibration_gate_20261106.json"
AUDIT_SAMPLE = "results/human_audit_sample_20261106.json"
AUDIT_ANALYSIS = "results/human_audit_analysis.json"


def in_domain_audit(root: Path) -> dict:
    """The blind audit counts only if it verifies against the gate registered before labelling."""
    analysis = json.loads(artifact_path(root, AUDIT_ANALYSIS).read_text(encoding="utf8"))
    gate = json.loads(artifact_path(root, GATE).read_text(encoding="utf8"))
    sample = json.loads(artifact_path(root, AUDIT_SAMPLE).read_text(encoding="utf8"))
    if gate["sample_sha256"] != sample["sample_sha256"]:
        raise ValueError("Gate and sample disagree")
    if analysis["gate"]["sha256"] != hashlib.sha256(artifact_path(root, GATE).read_bytes()).hexdigest():
        raise ValueError("Audit analysis cites another gate")
    if analysis["sample"]["sha256"] != sample["sample_sha256"]:
        raise ValueError("Audit analysis covers another sample")
    labels = artifact_path(root, analysis["labels"]["path"])
    if hashlib.sha256(labels.read_bytes()).hexdigest() != analysis["labels"]["sha256"]:
        raise ValueError("Audit labels changed after analysis")
    if analysis["sample"]["usable"] < gate["minimum_labelled_items"]:
        raise ValueError("Audit below the registered minimum")
    return analysis


def calibration_claim(root: Path) -> dict:
    try:
        analysis = in_domain_audit(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if (root / AUDIT_ANALYSIS).exists():
            return {"id": "judge-calibration", "status": "invalid",
                    "sources": [GATE, AUDIT_SAMPLE, AUDIT_ANALYSIS],
                    "reason": f"Human audit failed its own integrity checks: {exc}"}
        return {"id": "judge-calibration", "status": "pending", "sources": [GATE, AUDIT_SAMPLE],
                "reason": "Blind 180-item sample and a gate registered before labelling exist; labels not collected."}
    panel = analysis["panel"]["overall"]
    sens, spec = panel["sensitivity"], panel["specificity"]
    detail = (f"sensitivity {sens['count']}/{sens['n']} = {sens['rate']:.3f} "
              f"[{sens['wilson_ci95'][0]:.3f}, {sens['wilson_ci95'][1]:.3f}], "
              f"specificity {spec['count']}/{spec['n']} = {spec['rate']:.3f} "
              f"[{spec['wilson_ci95'][0]:.3f}, {spec['wilson_ci95'][1]:.3f}], "
              f"usable labels {analysis['sample']['usable']}")
    status = "evaluated" if analysis["gate_passed"] else "invalid"
    tail = ("Panel clears the pre-registered floors on this project's own generations; this is in-domain "
            "calibration only, not validity on other models or languages." if analysis["gate_passed"] else
            "Panel misses the pre-registered floors; the measured limits stand as the published limitation.")
    return {"id": "judge-calibration", "status": status,
            "sources": [GATE, AUDIT_SAMPLE, AUDIT_ANALYSIS], "reason": f"{detail}. {tail}"}

def build_readiness(root: Path, bundle: dict, verification: dict) -> dict:
    """Conservative claim ledger: presence is never scientific sign-off."""
    heldout = "results/p24_heldout_20261103_021508.json"
    heldout_judges = "results/p19_safety_judges_20261103_023754.json"
    state_checks = "results/p24_heldout_20261103_021508_state_checks.json"
    claims = [
        calibration_claim(root),
        {"id": "exploratory-comparison", "status": "evaluated", "sources": list(INPUTS.values()),
         "reason": "Full generations and judgments are replayable; failed calibration prevents confirmation or admission."},
        {"id": "safety-repair-confirmation", "status": "invalid", "sources": list(INPUTS.values()),
         "reason": "The bundled exploratory comparison is not evidence of production safety or repair equivalence."},
        {"id": "heldout-comparison", "status": "evaluated",
         "sources": [heldout, heldout_judges, state_checks, "results/heldout_analysis_20261106.json"],
         "reason": "Untouched prompts, already-judged completions, split disjointness and paired estimates are recorded in experiments/analyse_heldout.py. Still exploratory: the judge panel failed calibration, so this cannot confirm a repair claim."},
        {"id": "circuit-intervention", "status": "pending",
         "sources": ["results/circuit_body_recovery_20261106.json",
                     "results/p23_circuit_20261103_022433.json", "results/p26_circuit_arm_20261103_025214.json"],
         "reason": "The patch body is now recovered from the historical volume and hashes to the recorded value (150,994,944 scalars, 36 o_proj modules); no generation, judging or capability evaluation has been run on it, so the intervention claim stays pending."},
        {"id": "localisation-result", "status": "evaluated",
         "sources": ["results/localisation_analysis_20261106.json",
                     "results/p23_circuit_20261103_022433.json", "results/p1_grid.json",
                     "results/p2_dose.json"],
         "reason": "Direction-level localisation is measured (AUC 0.9988 to 0.8306 after ablation); the same analysis reports that circuit-node selection is NOT selective at a matched edit budget (0.000 difference against both blind controls). Exploratory re-analysis, reported as such."},
        {"id": "tamper-result", "status": "evaluated",
         "sources": ["results/tamper_analysis_20261106.json", "results/p2_dose.json",
                     "results/p3b_sweep.json", "results/p3c_blind.json"],
         "reason": "Removal and detection are measured on stored runs: a single direction cuts the refusal marker by 0.542 at no measured GSM8K cost, direction-aligned tampering at 10 % support is visible (+0.317 over its blind control) while blind tampering of the same size is not. Exploratory: refusal marker, uncalibrated panel."},
        {"id": "multilingual-confirmation", "status": "invalid",
         "sources": ["results/p29_multilingual_20261103_025940.json"],
         "reason": "Legacy non-finite projection results remain withdrawn; unfinished replacement output is not a result."},
        {"id": "project-human-audit", "status": "pending", "sources": list(INPUTS.values()),
         "reason": "Human annotation and adjudication on project completions have not been recorded."},
        {"id": "publication", "status": "pending", "sources": ["results/manifest.json"],
         "reason": "Local software does not establish public release, rebuilt report or license/security sign-off."},
    ]
    paths = {a["path"] for a in verification["artifacts"]}
    paths.update(path for claim in claims for path in claim["sources"])
    artifacts = []
    for relative in sorted(paths):
        item = {"path": relative, "status": "pending", "evaluation": "not independently validated"}
        try:
            raw = artifact_path(root, relative).read_bytes()
            item.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), available=True)
            if relative.endswith(".json"):
                data = json.loads(raw)
                if isinstance(data, dict):
                    item["recorded"] = {k: data[k] for k in
                                        ("status", "calibration_status", "headline_eligible", "passed") if k in data}
            if relative in INPUTS.values() and verification["status"] == "passed":
                item.update(status="evaluated", evaluation="Rebuilt and hash-linked; scientific eligibility remains separate")
        except (OSError, ValueError, TypeError) as exc:
            item.update(available=False, error=str(exc))
        artifacts.append(item)
    unavailable = {a["path"] for a in artifacts if not a["available"]}
    for claim in claims:
        missing = [p for p in claim["sources"] if p in unavailable]
        if missing:
            claim.update(status="pending",
                         reason=claim["reason"] + f" Source files unavailable here: {missing}.")
    if verification["status"] != "passed":
        claims[0].update(status="pending", reason="Current evidence integrity failed; do not use the comparison.")
    return {"schema": "scartissue.readiness/1", "checked_at": verification["checked_at"],
            "status": "not_submission_ready", "submission_ready": False,
            "integrity_status": verification["status"], "study_quality": bundle["quality"],
            "claims": claims, "artifacts": artifacts,
            "blockers": verification["scientific_blockers"],
            "status_definitions": {"valid": "Explicitly scoped sign-off, never inferred from a file existing",
                                   "evaluated": "An evaluation exists; it may be exploratory or have failed calibration",
                                   "pending": "Audit, execution or release sign-off remains incomplete",
                                   "invalid": "This claim cannot be made from the current evidence"},
            "limitations": verification["limitations"]}


def verify(root: Path = ROOT, bundle: dict | None = None) -> dict:
    checks = []
    artifacts = []
    blockers = []

    def check(name: str, passed: bool, detail):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    try:
        manifest = json.loads((root / "results/manifest.json").read_text(encoding="utf8"))
        check("manifest schema", manifest.get("schema") == "scartissue.manifest/1"
              and isinstance(manifest.get("evidence"), dict) and bool(manifest.get("model")),
              "Required schema, model and evidence map; additional keys are allowed")
        for name, relative in manifest.get("evidence", {}).items():
            try:
                path = artifact_path(root, relative)
                raw = path.read_bytes()
                if path.suffix == ".json":
                    json.loads(raw)
                artifacts.append({"name": name, "path": relative,
                                  "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                                  "status": "present_not_independently_validated"})
                check(f"manifest artifact: {name}", relative != "results/manifest.json", relative)
            except (OSError, ValueError, TypeError) as exc:
                check(f"manifest artifact: {name}", False, str(exc))
        check("manifest source coverage", set(INPUTS.values()).issubset(set(manifest.get("evidence", {}).values())),
              "Generation, judgments, safety analysis and lm-eval capability must be linked")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        manifest = {}
        check("manifest readable", False, str(exc))

    try:
        rebuilt = build_bundle(root)
        candidate = bundle if bundle is not None else json.loads(
            (root / "results/workbench_evidence_20261106.json").read_text(encoding="utf8"))
        check("bundle matches current evidence", comparable(candidate) == comparable(rebuilt),
              "All content and declared source hashes must match; creation time is not scientific evidence")
        validator = subprocess.run(
            ["node", "-e", "const fs=require('fs'); const E=require(process.argv[1]); E.validate(JSON.parse(fs.readFileSync(0,'utf8')));",
             str(root / "space-static/evidence.js")], input=json.dumps(candidate, allow_nan=False),
            capture_output=True, text=True, timeout=30)
        check("shared bundle schema", validator.returncode == 0, validator.stderr.strip())
        for source in candidate["sources"]:
            raw = artifact_path(root, source["path"]).read_bytes()
            check(f"bundle source hash: {source['path']}",
                  hashlib.sha256(raw).hexdigest() == source["sha256"], source["path"])
        for relative in ("space-static/data/evidence.json",):
            path = root / relative
            if path.exists():
                published = json.loads(path.read_text(encoding="utf8"))
                check(f"materialized bundle current: {relative}", comparable(published) == comparable(rebuilt), relative)
        quality = rebuilt["quality"]
        if quality.get("calibration_status") != "passed":
            blockers.append("Judge calibration has not passed")
        if quality.get("headline_eligible") is not True:
            blockers.append("Evidence is not headline eligible")
        audit = calibration_claim(root)
        if quality.get("human_project_audit") != "passed":
            blockers.append("Project-distribution human audit is pending")
        if audit["status"] == "pending":
            blockers.append("In-domain judge calibration has not been measured on human labels")
        elif audit["status"] == "evaluated":
            blockers.append("Panel is calibrated in-domain, but the frozen judge payload still records "
                            "calibration_status failed: eligibility needs a confirmatory re-judging run, not a "
                            "relabelled historical record")
        if quality.get("scope") != "heldout":
            blockers.append("Comparison is exploratory, not confirmatory")
        if quality.get("errors"):
            blockers.append("Evidence records execution or judgment errors")
        check("manifest eligibility agrees with sources",
              manifest.get("headline_eligible") == quality["headline_eligible"]
              and manifest.get("calibration_status") == quality["calibration_status"], quality)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, subprocess.SubprocessError) as exc:
        check("bundle reconstruction", False, str(exc))
        blockers.append("Bundle integrity or reconstruction is unresolved")

    # Release readiness requires separate evaluated-artifact and publication sign-off.
    blockers.append("Circuit-body evaluation, report rebuild and publication require separate sign-off")
    failed = sum(not item["passed"] for item in checks)
    return {"schema": "scartissue.verification/2", "checked_at": datetime.now(timezone.utc).isoformat(),
            "execution": "offline; no network or model inference", "n_checks": len(checks), "n_failed": failed,
            "status": "passed" if not failed else "failed", "submission_ready": False,
            "scientific_blockers": blockers, "checks": checks, "artifacts": artifacts,
            "limitations": ["Hashes establish internal consistency, not authenticity or scientific validity.",
                            "A passing integrity check cannot certify safety or submission readiness.",
                            "Historical PDF and public endpoints are not validated by this check."]}


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "passed" else 1)
