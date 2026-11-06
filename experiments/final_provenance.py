"""Build the final provenance bundle: manifest.json (15 keys), lexsi_provenance.json, and a
corrected corpus-provenance record (the 2026-11-02 file has a per-ID split_counts bug and is kept as
historical evidence, not overwritten).

    uv run python -m experiments.final_provenance
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "1-curate" / "corpus" / "corpus.jsonl"
SPLIT = ROOT / "1-curate" / "corpus" / "corpus_split.json"
RESULTS = ROOT / "results"

FILES = {
    "corpus": CORPUS,
    "split": SPLIT,
    "patches": ROOT / "artifacts_hf_local" / "manifest.json",
    "capability": RESULTS / "p16_capability_analysis_20261102.json",
    "safety_generations": RESULTS / "p21_safety_length_full_20261103_001722.json",
    "judge_verdicts": RESULTS / "p19_safety_judges_20261103_002850.json",
    "safety_analysis": RESULTS / "p21_length_analysis_20261103.json",
    "judge_validation": RESULTS / "p22_judge_validation_20261103_020927.json",
    "circuit": RESULTS / "p23_circuit_20261103_022433.json",
    "toolkits": RESULTS / "toolkit_provenance.json",
    "parquet": RESULTS / "parquet_publish_manifest.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    role_counts = Counter(split["by_id"][r["id"]] for r in rows)
    label_by_role: dict[str, Counter] = {}
    for r in rows:
        role = split["by_id"][r["id"]]
        label_by_role.setdefault(role, Counter())[r["label"]] += 1
    toolkits = json.loads(FILES["toolkits"].read_text(encoding="utf-8"))
    patches = json.loads(FILES["patches"].read_text(encoding="utf-8"))
    # The export manifest predates hash recording, so any null sha256 is filled from the
    # sha256 of the PUBLISHED .npz bytes (results/patch_checksums.json).
    checksums = {}
    cs_path = RESULTS / "patch_checksums.json"
    if cs_path.exists():
        checksums = {k: v.get("sha256_published")
                     for k, v in json.loads(cs_path.read_text(encoding="utf-8"))["patches"].items()}

    provenance = {
        "schema": "scartissue.provenance/1",
        "generated_at": "2026-11-03",
        "supersedes": "results/corpus_provenance_20261102.json (split_counts keyed by per-ID string "
                      "length; corrected here, historical file retained unedited)",
        "corpus": {
            "rows": len(rows),
            "unique_ids": len({r["id"] for r in rows}),
            "sources": dict(Counter(r["source_tag"] for r in rows)),
            "labels": dict(Counter(r["label"] for r in rows)),
            "roles": dict(role_counts),
            "labels_by_role": {k: dict(v) for k, v in label_by_role.items()},
            "every_row_has_id": all(r.get("id") for r in rows),
            "every_row_has_chain": all(r.get("provenance_chain") for r in rows),
            "frozen_split": {
                "N_MAL": split.get("N_MAL"), "N_SAFE": split.get("N_SAFE"),
                "N_FIT": split.get("N_FIT"), "slicing": split.get("slicing"),
                "is_partition": len(split["by_id"]) == len(rows),
                "counts": split.get("counts"),
            },
            "partitions_neve_reassigned_after_outcomes": True,
        },
        "models": {
            "headline": {"id": "CohereLabs/tiny-aya-global", "params_billion": 3.35,
                         "architecture": "cohere2", "layers": 36,
                         "revision": "af89d219b53ed9b13b8a4645f9c8028973510324"},
            "secondary": ["CohereLabs/tiny-aya-earth", "CohereLabs/tiny-aya-water"],
            "judges": [
                {"role": "harmbench", "id": "cais/HarmBench-Mistral-7b-val-cls",
                 "revision": "51182c7cdaf9b2e5f05b745b81b047b67e7384a1"},
                {"role": "qwen3guard", "id": "Qwen/Qwen3Guard-Gen-8B"},
                {"role": "granite", "id": "ibm-granite/granite-guardian-3.2-5b"},
            ],
        },
        "patches": {
            "files": sorted(patches.get("arms", patches.get("patches", {})).keys()),
            "sha256": {k: ((v.get("sha256") if isinstance(v, dict) else None) or checksums.get(k))
                       for k, v in patches.get("arms", patches.get("patches", {})).items()},
            "sha256_source": "results/patch_checksums.json (hash of the published .npz bytes); "
                             "null only where the patch body is not published",
            "note": "factorised parameters + explicit allowed{modules,layers,heads}; no drifted "
                    "checkpoint is published (LSAL 1.2/1.1 §4)",
        },
        "toolkits": toolkits.get("track_repos_used_together", toolkits),
        "inputs_sha256": {k: sha256(p) for k, p in FILES.items() if p.exists()},
        "missing_inputs": [k for k, p in FILES.items() if not p.exists()],
    }
    (RESULTS / "provenance_final.json").write_text(
        json.dumps(provenance, indent=1), encoding="utf-8")

    manifest = {
        "schema": "scartissue.manifest/1",
        "title": "ScarTissue — abliteration removes a read-out; support size sets the repair price",
        "track": "Lexsi AI Safety & Alignment Hackathon, Track 2a",
        "date": "2026-11-03",
        "claim": "Abliteration deletes the refusal read-out and the discrimination behind it; a "
                 "weight-only repair recovers refusal, and the capability price is set by the "
                 "support of the edit, not by the model's breadth of refusal.",
        "model": "CohereLabs/tiny-aya-global @ af89d219",
        "track_repos": toolkits.get("track_repos_used_together", []),
        "datasets": {
            "corpus": "amugoodbad229/scartissue-tinyaya-safety-corpus",
            "results": "amugoodbad229/scartissue-tinyaya-results",
            "drift": "amugoodbad229/scartissue-tinyaya-abliteration-patches",
            "repair": "amugoodbad229/scartissue-tinyaya-repair-patches",
        },
        "deployment": "https://huggingface.co/spaces/amugoodbad229/scartissue",
        "metrics": ["refusal restoration", "benign over-refusal", "harm-ASR (judge majority-3)",
                    "GSM8K exact match (lm-eval)"],
        "headline_numbers": {
            "base": {"refusal": 0.842, "over_refusal": 0.075, "harm_asr": 0.058, "gsm8k": 0.590},
            "drifted": {"refusal": 0.500, "over_refusal": 0.050, "harm_asr": 0.217, "gsm8k": 0.560},
            "harm2_a4": {"refusal": 0.883, "over_refusal": 0.175, "harm_asr": 0.050, "gsm8k": 0.518},
            "global_a4": {"refusal": 0.883, "over_refusal": 0.275, "harm_asr": 0.042, "gsm8k": 0.438},
            "capability_gap_harm2_minus_global": {"delta": 0.080, "ci95": [0.038, 0.122]},
        },
        "evidence": {k: f"results/{p.name}" for k, p in FILES.items() if p.exists()},
        "judge_calibration": {"n": 500, "source": "nvidia/Aegis-AI-Content-Safety-Dataset-1.0",
                              "majority": {"accuracy": 0.782, "sensitivity": 0.580,
                                           "specificity": 0.984}},
        "limitations": [
            "one model, one seed, one capability task carry the headline",
            "harm-ASR absolute level is a lower bound (panel specificity 0.984, sensitivity 0.580)",
            "circuit arm exported but not yet evaluated (compute blocked)",
            "held-out and multilingual runs generated/queued, not judged",
        ],
        "licence": "Cohere LSAL 1.2/1.1, research use; parameters published, no drifted checkpoint",
        "provenance": "results/provenance_final.json",
        "report": "report/report.pdf",
    }
    assert len(manifest) == 17, f"manifest key count changed: {len(manifest)}"
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    lexsi = {
        "schema": "scartissue.lexsi/1",
        "track": "2a",
        "hard_requirements": {
            "track_repos_used_together": toolkits.get("track_repos_used_together", []),
            "cohere_dense_open_weights": True,
            "hf_artifacts": True,
            "hf_artifacts_format": "parquet (corpus + results), .npz (parameter patches)",
            "public_deployment": "https://huggingface.co/spaces/amugoodbad229/scartissue",
            "more_than_one_before_after_metric": True,
            "writeup_rationale": "report/report.pdf, WRITE-UP.md §9",
        },
        "before_after": {
            "metric_1_refusal": {"before": 0.842, "drifted": 0.500, "repaired_2pct": 0.883},
            "metric_2_over_refusal": {"before": 0.075, "drifted": 0.050, "repaired_2pct": 0.175,
                                      "repaired_global": 0.275},
            "metric_3_harm_asr": {"before": 0.058, "drifted": 0.217, "repaired_2pct": 0.050},
            "metric_4_gsm8k": {"before": 0.590, "drifted": 0.560, "repaired_2pct": 0.518,
                               "repaired_global": 0.438},
        },
        "licence_compliance": {"deploy_drifted_checkpoint": False,
                               "sells_or_hosts_model_as_product": False,
                               "organisation_acknowledgement": "not sent (individual entry)"},
        "provenance": "results/provenance_final.json",
    }
    (RESULTS / "lexsi_provenance.json").write_text(json.dumps(lexsi, indent=1), encoding="utf-8")

    print("[ok] provenance_final.json, manifest.json, lexsi_provenance.json")
    print("[roles]", dict(role_counts))
    print("[toolkits]", toolkits.get("track_repos_used_together"))
    print("[missing]", provenance["missing_inputs"])


if __name__ == "__main__":
    main()
