"""Frozen corpus partitions, with no evaluation leakage.

Every fitting, discovery and held-out selection reads this module so that the 120
harmful + 40 benign evaluation prompts can never reappear in a fit or discovery set.
The partition file is the single source of truth (`1-curate/corpus/corpus_split.json`);
nothing here can reassign a role.

    fit_prompts()        -> (16 harmful, 16 benign)         the only prompts `d` is fitted on
    evaluation_rows()    -> the frozen 120 harmful + 40 benign rows
    heldout_rows(seed)   -> 120 harmful + 40 benign from `unused` (never used in any arm)
    discovery_pairs(...) -> token-length-matched fit harmful/benign pairs for circuit discovery
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

CORPUS = Path("1-curate/corpus")
EVAL_HARMFUL, EVAL_BENIGN = 120, 40


def _load():
    rows = [json.loads(line) for line in (CORPUS / "corpus.jsonl").read_text(encoding="utf8").splitlines() if line.strip()]
    split = json.loads((CORPUS / "corpus_split.json").read_text(encoding="utf8"))
    by_id = split["by_id"]
    if len(by_id) != len(rows) or {r["id"] for r in rows} != set(by_id):
        raise ValueError("corpus rows and frozen split disagree")
    counts = {role: sum(1 for v in by_id.values() if v == role) for role in set(by_id.values())}
    expected = {"eval_malicious": EVAL_HARMFUL, "eval_benign": EVAL_BENIGN,
                "fit_malicious": 16, "fit_benign": 16}
    if any(counts.get(k, 0) != v for k, v in expected.items()):
        raise ValueError(f"frozen partition changed: {counts}")
    return rows, by_id, split


def rows_for(role: str) -> list[dict]:
    rows, by_id, _ = _load()
    out = [{"id": r["id"], "role": role, "prompt": r["text"], "source_tag": r["source_tag"],
            "label": r["label"], "prompt_sha256": hashlib.sha256(r["text"].encode()).hexdigest()}
           for r in rows if by_id[r["id"]] == role]
    return sorted(out, key=lambda r: r["id"])


def fit_prompts() -> tuple[list[str], list[str]]:
    mal = [r["prompt"] for r in rows_for("fit_malicious")]
    ben = [r["prompt"] for r in rows_for("fit_benign")]
    if len(mal) != 16 or len(ben) != 16:
        raise ValueError("fit partition must be 16/16")
    return mal, ben


def evaluation_rows() -> list[dict]:
    rows = rows_for("eval_malicious") + rows_for("eval_benign")
    if len(rows) != EVAL_HARMFUL + EVAL_BENIGN:
        raise ValueError("evaluation partition must be 120 + 40")
    forbidden = {r["prompt"] for r in rows_for("fit_malicious") + rows_for("fit_benign")}
    if forbidden & {r["prompt"] for r in rows}:
        raise ValueError("fit/eval prompt overlap")
    return rows


def heldout_rows(seed: int = 20261103) -> list[dict]:
    """Untouched prompts: only rows the frozen split never assigned to fit or eval."""
    rows, by_id, _ = _load()
    evals = set(by_id[k] for k in by_id if by_id[k].startswith("eval_") or by_id[k].startswith("fit_"))
    pool_harm = [r for r in rows if by_id[r["id"]] == "unused" and r["label"] == "harmful" and r["id"] not in evals]
    pool_safe = [r for r in rows if by_id[r["id"]] == "unused" and r["label"] == "safe"]
    if len(pool_harm) < EVAL_HARMFUL or len(pool_safe) < EVAL_BENIGN:
        raise ValueError("insufficient unused rows for a held-out draw")
    rng = random.Random(seed)
    harm = rng.sample(sorted(pool_harm, key=lambda r: r["id"]), EVAL_HARMFUL)
    safe = rng.sample(sorted(pool_safe, key=lambda r: r["id"]), EVAL_BENIGN)
    out = []
    for role, group in (("heldout_malicious", harm), ("heldout_benign", safe)):
        for r in sorted(group, key=lambda r: r["id"]):
            out.append({"id": r["id"], "role": role, "prompt": r["text"], "source_tag": r["source_tag"],
                        "label": r["label"], "prompt_sha256": hashlib.sha256(r["text"].encode()).hexdigest()})
    return out


def discovery_pairs(token_len, n_pairs: int = 24) -> list[tuple[str, str]]:
    """Length-matched (harmful, benign) pairs from the FIT partition only."""
    mal, ben = fit_prompts()
    by_len: dict[int, list[str]] = {}
    for prompt in ben:
        by_len.setdefault(token_len(prompt), []).append(prompt)
    pairs = []
    for prompt in mal:
        bucket = by_len.get(token_len(prompt))
        if bucket:
            pairs.append((prompt, bucket.pop(0)))
    if len(pairs) < 6:
        raise ValueError(f"only {len(pairs)} length-matched fit pairs")
    return pairs[:n_pairs]
