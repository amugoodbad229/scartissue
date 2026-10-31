"""Two cheap gates that can invalidate the design before any building happens.

GATE A  token_pair   — is there a genuine single-token refusal/compliance pair in the 262k
                       CohereTokenizer? CircuitKIT's default (" I" / " Sure") is unverified for
                       this vocab (repo-maps/CircuitKIT.md §8.2). Needs no GPU of consequence.
GATE B  cohere_parity— does CircuitKIT's transformer-lens port of cohere2 reproduce HF logprobs?
                       (repo-maps/CircuitKIT.md §2: KL < 1e-4 on the final position, 4 fixed prompts,
                       plus a 4196-token sliding-window test). EXPERIMENTAL family, never run on
                       tiny-aya in CI. If this fails, the circuit leg is inert -> fall back to
                       Candidate 2 (repair-transfer matrix, no circuit dependency).

Run:  uv run modal run experiments/parity_gate.py
Out:  results/parity_gate.json   log: logs/parity_gate.log
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CK = ROOT / "repos" / "CircuitKIT"
OUT = ROOT / "results" / "parity_gate.json"

MODEL = "CohereLabs/tiny-aya-base"  # what the parity test gates on

# Candidate answer prefixes. The refusal axis needs BOTH to be one token: the score is
# logit(" I") - logit(" Sure") at the answer position.
REFUSAL_CANDIDATES = [
    " I", " Sorry", " Unfortunately", " No", " Cannot", " Î", " Sorry,", " I'm",
    "I", " As", " It", " That",
]
COMPLIANCE_CANDIDATES = [
    " Sure", " Here", " To", " The", " Yes", " Certainly", " Of", " First", " You", " Step",
]

img = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "pkg-config", "libicu-dev")
    .pip_install(
        "torch==2.9.0",
        "transformers>=5.15,<6",
        "huggingface-hub>=1.5,<2",
        "transformer-lens==3.8.0",
        "safetensors",
        "datasets",
        "accelerate",
        "einops",
        "plotly",
        "pandas",
        "scipy",
        "networkx",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "evaluate",
        "sacrebleu",
        "rouge_score",
        "bert-score",
        "sentence-transformers",
        "click",
        "rich",
        "spacy>=3.8",
        "psutil",
        "torch-pruning>=1.0",
        "ordered-set",
        "ipywidgets",
        "pyyaml",
        "pytest",
        "pytest-timeout",
    )
    # CircuitKIT is NOT pip-installed: its dynamic version needs a build backend that is not
    # worth debugging here. The tests only need the package importable, so put src/ on PYTHONPATH.
    .add_local_dir(str(CK), remote_path="/root/CircuitKIT", copy=True)
)
app = modal.App("scartissue-parity-gate")
_secret = modal.Secret.from_name("huggingface")
def _run_streaming(cmd, cwd, env, timeout_s):
    """Run a subprocess and STREAM its output to the Modal log as it arrives.

    Why: `subprocess.run(capture_output=True)` swallows everything until the process exits, so the
    dashboard shows a silent container for 20-60 minutes and a hang is indistinguishable from work.
    Here every line is printed immediately (prefixed) and also collected for the result payload.
    """
    import subprocess
    lines = []
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        lines.append(line)
        print("[test] " + line, flush=True)
    proc.wait(timeout=timeout_s)
    return proc.returncode, "\n".join(lines)




@app.function(image=img, gpu="A10G", timeout=900, secrets=[_secret])
def token_pair() -> str:
    import torch
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    out = {
        "model": MODEL,
        "vocab_size": len(tok),
        "add_bos_token": tok.add_bos_token,
        "has_chat_template": bool(getattr(tok, "chat_template", None)),
    }

    def one(s: str):
        ids = tok.encode(s, add_special_tokens=False)
        return {"text": s, "n_tokens": len(ids), "ids": ids[:6]}

    out["refusal"] = [one(s) for s in REFUSAL_CANDIDATES]
    out["compliance"] = [one(s) for s in COMPLIANCE_CANDIDATES]

    ok_r = [r["text"] for r in out["refusal"] if r["n_tokens"] == 1]
    ok_c = [r["text"] for r in out["compliance"] if r["n_tokens"] == 1]
    # default pair from CircuitKIT data/adapters/safety_prompt.py:55
    out["default_pair"] = {"refusal": one(" I"), "compliance": one(" Sure")}
    out["default_pair_usable"] = (
        out["default_pair"]["refusal"]["n_tokens"] == 1
        and out["default_pair"]["compliance"]["n_tokens"] == 1
    )
    out["single_token_refusal"] = ok_r
    out["single_token_compliance"] = ok_c
    out["usable_pairs"] = len(ok_r) * len(ok_c)
    out["verdict"] = (
        "PASS default pair is single-token"
        if out["default_pair_usable"]
        else f"SWAP needed: {len(ok_r)}x{len(ok_c)} alternatives exist"
    )
    print(json.dumps({k: out[k] for k in ("model", "default_pair_usable", "usable_pairs", "verdict")}, indent=2))
    return json.dumps(out)


@app.function(image=img, gpu="A100-40GB", timeout=5400, secrets=[_secret])
def cohere_parity() -> str:
    import subprocess
    import sys

    env = dict(os.environ)
    env["CIRCUITKIT_RUN_TINY_AYA"] = "1"
    env["HF_HOME"] = "/root/.cache/huggingface"
    env["PYTHONPATH"] = "/root/CircuitKIT/src"  # package is not pip-installed
    log = Path("/root/parity.log")

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/backends/tl_compat/test_cohere_parity.py",
        # pytest.ini sets addopts = -m "not slow and not gated and not network", and every test in
        # this file is marked slow + gated → without an explicit -m override all 10 are DESELECTED
        # and pytest exits 5 ("no tests ran"), which is not a statement about the port.
        "-m", "",
        "-v", "-x", "--no-header", "-p", "no:cacheprovider",
    ]
    rc, out = _run_streaming(cmd, "/root/CircuitKIT", env, 4800)
    log.write_text(out)
    return json.dumps({
        "rc": rc,
        "tail": "\n".join(out.strip().splitlines()[-40:]),
        "stderr_tail": "",
        "verdict": "PASS" if rc == 0 else "FAIL",
    })[:20000]


@app.function(image=img, gpu="A100-80GB", timeout=7200, secrets=[_secret])
def expanse_parity() -> str:
    """Aya Expanse 8B (HF arch CohereForCausalLM, port module `cohere`) — one of the SEVEN models the
    hackathon allows, so its parity result decides the mechanism leg independently of tiny-aya.
    fp32 both copies ~= 2 x 32 GB, hence A100-80GB."""
    import subprocess
    import sys

    env = dict(os.environ)
    env["CIRCUITKIT_RUN_AYA_EXPANSE"] = "1"
    env["HF_HOME"] = "/root/.cache/huggingface"
    env["PYTHONPATH"] = "/root/CircuitKIT/src"

    cmd = [
        sys.executable, "-m", "pytest",
        "tests/backends/tl_compat/test_aya_expanse_parity.py",
        "-m", "",  # pytest.ini's addopts deselects slow/gated/network; these tests are all three
        "-v", "-x", "--no-header", "-p", "no:cacheprovider",
    ]
    rc, out = _run_streaming(cmd, "/root/CircuitKIT", env, 6600)
    return json.dumps({
        "rc": rc,
        "tail": "\n".join(out.strip().splitlines()[-40:]),
        "stderr_tail": "",
        "verdict": "PASS" if rc == 0 else "FAIL",
    })[:20000]


@app.local_entrypoint()
def main(only: str = "both"):
    res: dict = {}
    if OUT.exists():  # merge, so --only runs do not clobber the other gate's result
        try:
            res = json.loads(OUT.read_text())
        except Exception:
            res = {}
    if only in ("both", "token"):
        print("== GATE A token_pair ==")
        res["token_pair"] = json.loads(token_pair.remote())
        print(res["token_pair"]["verdict"])
    if only in ("both", "parity"):
        print("== GATE B cohere_parity (A100-40GB, ~20 min) ==")
        got = json.loads(cohere_parity.remote())
        res["cohere_parity"] = got
        print(got["verdict"])
        print(got["tail"][-2000:])
    if only in ("both", "expanse"):
        print("== GATE B2 aya-expanse-8b parity (A100-80GB) ==")
        got = json.loads(expanse_parity.remote())
        res["expanse_parity"] = got
        print(got["verdict"])
        print(got["tail"][-2000:])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT}")
