"""Re-check patch application states for a generation run without re-generating.

`p24_heldout.py` compared the workers' ``edited_scalars`` map (keyed by patch FILE stem) against
``expected_states`` (keyed by ROLE: ``drift`` / ``repair``), so every non-base arm was marked failed
even though generation, hashes and resets all passed. The raw file keeps that audit trail; this
script recomputes the comparison with a role-aware mapping so the run can be judged on what it
actually measured.

Checks per arm:
  base_write_weights_sha256 == expected base hash          (state before any patch)
  write_weights_sha256      == expected post-patch hash    (state after the patches)
  edited scalars            -> drift scalars == expected drift, repair scalars == expected repair
  restored hash             == base hash                   (exact reset after generation)

Writes ``results/<run>_state_checks.json`` next to the run; never mutates the run.

    uv run python -m experiments.verify_states results/p24_heldout_20261103_021508.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIFT_SCALARS = 962_592_768


def role_of(stem: str) -> str:
    return "drift" if stem.startswith("abliteration") else "repair"


def main(argv: list[str]) -> None:
    path = Path(argv[0])
    payload = json.loads(path.read_text(encoding="utf8"))
    expected = payload["expected_states"]
    report = {"schema": "scartissue.state-checks/1", "run": str(path).replace("\\", "/"),
              "note": ("Role-aware re-check of the original run's patch application. Generation "
                       "records, hashes and resets are untouched; only the key mapping the original "
                       "checker got wrong is corrected."),
              "arms": {}, "passed": True}
    for worker in payload["workers"]:
        arm = worker["arm"]
        want = expected[arm]
        have_scalars = worker.get("edited_scalars") or {}
        by_role = {"drift": 0, "repair": 0}
        for stem, value in have_scalars.items():
            by_role[role_of(stem)] += int(value)
        checks = {
            "base_state_hash": (worker.get("base_write_weights_sha256"), want["base_hash"]),
            "post_patch_state_hash": (worker.get("write_weights_sha256"), want["write_weights_sha256"]),
            "drift_scalars": (by_role["drift"], int(want["edited_scalars"]["drift"])),
            "repair_scalars": (by_role["repair"], int(want["edited_scalars"]["repair"])),
            "reset_hash": (worker.get("restored_base_write_weights_sha256"), want["base_hash"]),
            "record_count": (len(worker.get("records", [])), len(payload["rows"])),
        }
        failed = {k: {"have": v[0], "want": v[1]} for k, v in checks.items() if v[0] != v[1]}
        report["arms"][arm] = {"passed": not failed, "failures": failed,
                               "edited_scalars_by_role": by_role}
        report["passed"] = report["passed"] and not failed
    out = path.with_name(path.stem + "_state_checks.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf8")
    for arm, entry in report["arms"].items():
        print(f"{arm:10s} {'PASS' if entry['passed'] else 'FAIL'}  drift={entry['edited_scalars_by_role']['drift']} "
              f"repair={entry['edited_scalars_by_role']['repair']}"
              + ("" if entry["passed"] else f"  {entry['failures']}"))
    print(("ALL ARMS PASS" if report["passed"] else "FAILURES REMAIN"), out)


if __name__ == "__main__":
    main(sys.argv[1:])
