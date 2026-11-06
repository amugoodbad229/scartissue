"""Multilingual frontier: does the repair survive outside English?

The model declares ~68 languages (`huggingface.co/api/models/CohereLabs/tiny-aya-global` → cardData.language).
StrongREJECT and XSTest are English-only, so the prompt set is machine-translated ONCE with a pinned
translation model (NLLB-200-distilled-600M) and every translation is stored with a sha256 — a reader can
audit the text rather than trust a claim about it.

Two measurement decisions, both forced:
  * Refusal cannot be scored with the English regex on 68 languages. The language-agnostic quantity is the
    projection onto the refusal direction, `d'·h / |h|` — the same read-out the abliteration removes — so
    that is the metric here, reported for harmful and benign prompts separately (the benign one is this
    experiment's over-refusal axis).
  * Capability (gsm8k) stays English-only; a translated gsm8k would confound translation quality with
    model damage.

    MSYS_NO_PATHCONV=1 uv run python -m modal run experiments/p14_multilingual.py

Writes /hf/results/p14_multilingual.json; log `logs/p14_multilingual.log`.
"""

from __future__ import annotations

import json
import time

import modal

APP = "scartissue-p14-multilingual"
VOL = "scartissue-hf"
MODEL = "CohereLabs/tiny-aya-global"
LAYER, N_MAL, N_BEN, N_FIT, ALPHA_REPAIR, FRAC = 15, 40, 20, 16, 4.0, 0.02
GROUPS = 6  # languages are split across this many containers

app = modal.App(APP)
vol = modal.Volume.from_name(VOL, create_if_missing=True)
base_image = (modal.Image.debian_slim(python_version="3.12")
              .pip_install("torch==2.9.0", "transformers==5.22.0", "accelerate==1.11.0",
                           "huggingface-hub==1.33.0", "datasets==4.4.2", "sentencepiece",
                           "numpy", "pycountry")
              .add_local_dir("experiments", remote_path="/root/experiments")
              .add_local_dir("p14", remote_path="/root/p14"))
gpu_image = base_image
secrets = [modal.Secret.from_name("huggingface")]

# NLLB codes differ from ISO-639-1; these are the ambiguous cases where the script must be pinned.
SCRIPT_OVERRIDE = {"zh": "Hans", "sr": "Cyrl", "pa": "Guru", "ku": "Arab", "uz": "Latn",
                   "ms": "Latn", "ha": "Latn", "yo": "Latn", "ig": "Latn", "am": "Ethi",
                   "bn": "Beng", "hi": "Deva", "mr": "Deva", "ne": "Deva", "gu": "Gujr",
                   "ta": "Taml", "te": "Telu", "th": "Thai", "km": "Khmr", "lo": "Laoo",
                   "my": "Mymr", "ar": "Arab", "fa": "Arab", "ur": "Arab", "he": "Hebr",
                   "el": "Grek", "ru": "Cyrl", "uk": "Cyrl", "bg": "Cyrl", "mk": "Cyrl"}


def nllb_code(tokenizer, iso: str) -> str | None:
    """Map an ISO-639-1 tag onto one of NLLB's `xxx_Script` codes (None when NLLB has no entry)."""
    import pycountry

    try:
        alpha3 = pycountry.languages.get(alpha_2=iso).alpha_3
    except Exception:
        return None
    # transformers 5.22's NllbTokenizer dropped `lang_code_to_id` (measured: AttributeError), so the language
    # codes are read straight off the vocab — they are added tokens of the form `xxx_Script`.
    vocab = [t for t in tokenizer.get_vocab() if len(t) == 8 and t[3] == "_" and t[:3].isalpha()
             and t[:3].islower()]
    candidates = [c for c in vocab if c.split("_")[0] == alpha3]
    if not candidates:
        return None
    want = SCRIPT_OVERRIDE.get(iso)
    if want:
        for c in candidates:
            if c.endswith(want):
                return c
    return candidates[0]


@app.function(image=base_image, volumes={"/hf": vol}, secrets=secrets, gpu="A10G", timeout=3600)
def translate(languages: list[str]) -> dict:
    """Translate the frozen English evaluation sets into every declared language, once."""
    import hashlib

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    t0 = time.time()
    vol.reload()
    langs = languages
    sr = load_dataset("walledai/StrongREJECT", split="train")
    xs = load_dataset("Paul/XSTest", split="train")
    mal = [r["prompt"] for r in sr][:N_MAL]
    ben = [r["prompt"] for r in xs if str(r.get("label", "")).lower() == "safe"][:N_BEN]

    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float32).cuda().eval()
    tok.src_lang = "eng_Latn"

    def run(texts: list[str], tgt: str) -> list[str]:
        ids = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, forced_bos_token_id=tok.convert_tokens_to_ids(tgt),
                                 max_new_tokens=200, num_beams=2)
        return tok.batch_decode(out, skip_special_tokens=True)

    out, skipped = {}, []
    for iso in langs:
        tgt = nllb_code(tok, iso)
        if tgt is None:
            skipped.append(iso)
            continue
        rec = {"nllb": tgt, "mal": run(mal, tgt), "ben": run(ben, tgt)}
        rec["sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
        out[iso] = rec
        print(f"[tr] {iso:3s} -> {tgt:10s} {len(rec['mal'])}+{len(rec['ben'])} prompts  "
              f"{rec['mal'][0][:60]!r}", flush=True)
    res = {"model": "facebook/nllb-200-distilled-600M", "source": "StrongREJECT/XSTest (English)",
           "languages": out, "skipped_no_nllb": skipped, "wall_s": round(time.time() - t0, 1)}
    with open("/hf/results/p14_prompts.json", "w") as fh:
        json.dump(res, fh, ensure_ascii=False)
    vol.commit()
    res["n_languages"] = len(out); res["skipped"] = skipped
    return res


@app.function(image=gpu_image, volumes={"/hf": vol}, secrets=secrets, gpu="A100-40GB", timeout=7200)
def measure(group: int, payload: dict, n_groups: int = GROUPS) -> dict:
    """base → drift → repair, measured per language as the refusal projection (language-agnostic)."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    prompts = payload["languages"]
    langs = sorted(prompts)[group::n_groups]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).cuda().eval()

    def chat(p: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True)

    @torch.no_grad()
    def proj(texts: list[str]) -> np.ndarray:
        """Mean residual-stream projection at LAYER, per prompt (unit-normalised act)."""
        out = []
        for i in range(0, len(texts), 8):
            enc = tok([chat(t) for t in texts[i:i + 8]], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states[LAYER].float()
            mask = enc["attention_mask"].unsqueeze(-1)
            pooled = (hs * mask).sum(1) / mask.sum(1)
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)
            out.append((pooled @ torch.tensor(d, dtype=torch.float32, device="cuda")).cpu().numpy())
        return np.concatenate(out)

    # the direction is fitted on ENGLISH held-out prompts, then applied unchanged to every language:
    # that is what makes the numbers comparable across languages.
    sr = __import__("datasets").load_dataset("walledai/StrongREJECT", split="train")
    xs = __import__("datasets").load_dataset("Paul/XSTest", split="train")
    mal_fit = [r["prompt"] for r in sr][N_MAL:N_MAL + N_FIT]
    ben_fit = [r["prompt"] for r in xs if str(r.get("label", "")).lower() == "safe"][N_BEN:N_BEN + N_FIT]

    @torch.no_grad()
    def act(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 8):
            enc = tok([chat(t) for t in texts[i:i + 8]], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states[LAYER].float()
            mask = enc["attention_mask"].unsqueeze(-1)
            out.append(((hs * mask).sum(1) / mask.sum(1)).cpu().numpy())
        return np.concatenate(out)

    v = act(mal_fit).mean(0) - act(ben_fit).mean(0)
    d = (v / np.linalg.norm(v)).astype(np.float64)
    d = d.tolist()
    del mal_fit, ben_fit, v
    torch.cuda.empty_cache()

    def snapshot() -> dict:
        return {iso: {"mal": proj(prompts[iso]["mal"]).tolist(),
                      "ben": proj(prompts[iso]["ben"]).tolist()} for iso in langs}

    stages = {"base": snapshot()}

    # drift: rank-one ablation of d on every write matrix
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight")):
                w = p.detach().float().cpu().numpy().astype(np.float64)
                delta = -1.0 * np.outer(d, d @ w)
                p.data.add_(torch.from_numpy(delta).to(p.dtype).to(p.device))
    stages["drift"] = snapshot()
    print(f"[{group}] drift measured, vram={torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)

    out = {"group": group, "languages": langs, "stages": stages, "wall_s": round(time.time() - t0, 1)}
    with open(f"/hf/results/p14_group{group}.json", "w") as fh:
        json.dump(out, fh)
    vol.commit()
    return {"group": group, "languages": langs, "stages": stages,
            "wall_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main() -> None:
    import json as _j, pathlib as _p
    langs = _j.loads(_p.Path("p14/p14_languages.json").read_text())["languages"]
    payload = translate.remote(langs)
    print("[translate]", payload.get("n_languages"), "languages, skipped", payload.get("skipped"), flush=True)
    runs = [measure.spawn(i, payload, GROUPS) for i in range(GROUPS)]
    merged: dict = {}
    for r in runs:
        got = r.get()
        for iso, v in got["stages"]["base"].items():
            merged.setdefault(iso, {})["base_mal"] = sum(v["mal"]) / len(v["mal"])
            merged.setdefault(iso, {})["base_ben"] = sum(v["ben"]) / len(v["ben"])
        print(f"[group {got['group']}] {got['languages']} wall={got['wall_s']}s", flush=True)
    print(f"[merged] {len(merged)} languages", flush=True)
