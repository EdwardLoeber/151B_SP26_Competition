#!/usr/bin/env python3
"""
run_inference.py — single end-to-end entry point for the CSE 151B competition.

This reproduces, in one function call, the exact two-stage pipeline that produced
our Kaggle submission. It is a self-contained copy of the logic in
`eddie/self_consistency.ipynb` (generation) and `eddie/Post_processing.ipynb`
(aggregation, the "v2" per-slot vote). The original notebooks are left untouched;
this script only imports the official grader (`judger.py`) from the `eddie/`
folder and writes all of its own output under `outputs/tester/`.

Pipeline
--------
Stage 1 — Self-consistency generation (eddie/self_consistency.ipynb §4–§7):
    Sample N=7 completions per private question from the *untouched base model*
    Qwen/Qwen3-4B-Thinking-2507 (temp 0.6) and archive every raw sample.
    Resumable: re-running skips questions already in the archive.

Stage 2 — Post-processing v2 aggregation (eddie/Post_processing.ipynb):
    For each question, parse the \\boxed{} sub-answers of every sample, vote each
    [ANS] slot independently using judger-equivalence clustering, rebuild the
    final answer, and write the submission CSV.

We did NOT fine-tune for the final submission (GRPO came in at 70.41% vs the
71.60% base), so there are no custom weights to download — the base model loads
straight from the Hugging Face Hub by ID. Requirement #3 (uploading fine-tuned
weights) therefore does not apply.

Hardware / time
---------------
Stage 1 ran on a single A100-80GB (Google Colab) and took ~20 hours for the full
private set at N=7 with the full 32,768-token thinking budget. Stage 2 is CPU-only
and takes a few minutes. An H100 is ~1.6x faster.

Usage
-----
    from run_inference import run_inference
    run_inference()                                  # uses ./data/private.jsonl
    run_inference(private_path="/path/to/private.jsonl")

or from the command line:

    python run_inference.py --private-path data/private.jsonl

The result is written to outputs/tester/tester_submission.csv (and the raw
per-sample archive to outputs/tester/tester__private__samples.jsonl).
"""

import os
import re
import csv
import sys
import json
from collections import Counter

# ──────────────────────────────────────────────────────────────────────────────
# Paths & config (all hyperparameters are the FINAL ones used for the submission)
# ──────────────────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
EDDIE_DIR = os.path.join(HERE, "eddie")          # source of the official judger.py / utils.py

# Import the OFFICIAL grader from eddie/ (so equivalence clustering matches the
# grader exactly). Inserted at the front so eddie/judger.py wins over any copy
# sitting next to this script. judger.py does `from utils import *`, so eddie/
# must be on the path for utils.py too.
if EDDIE_DIR not in sys.path:
    sys.path.insert(0, EDDIE_DIR)

# Model — the untouched base model (no fine-tuned weights to download).
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
MODEL_TAG = "tester"                              # namespaces / labels every output file

# Sampling (identical to self_consistency.ipynb §3 + §6).
N_SAMPLES      = 7            # odd, so majority votes stay decisive
TEMPERATURE    = 0.6         # Qwen3-recommended; diversity comes from sampling
TOP_P          = 0.95
TOP_K          = 20
MAX_GEN_TOKENS = 32768       # full thinking budget at inference
MAX_MODEL_LEN  = 40960
GPU_MEM_UTIL   = 0.90
SEED           = 151
CHUNK          = 40          # problems per generate() call; checkpoint after each (resumable)

# Default I/O. The private set is provided by the graders; default to ./data/private.jsonl.
DEFAULT_PRIVATE_PATH = os.path.join(HERE, "data", "private.jsonl")
DEFAULT_OUTPUT_DIR   = os.path.join(HERE, "outputs", "tester")

# Prompts (identical to Phase-1 baseline / eval — self_consistency.ipynb §4).
SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Give your final answer inside a single \\boxed{}. "
    "Use EXACT values: prefer fractions (\\frac{a}{b}) and symbolic forms "
    "(\\sqrt{}, \\pi, e) over decimals. If you must give a decimal, write at "
    "least 10 significant figures and do NOT round. "
    "If the problem has multiple sub-answers, put them all inside one \\boxed{}, "
    "comma-separated, in the order asked, e.g. \\boxed{41, 35, 16}. "
    "If a single sub-answer itself contains a comma (a point or tuple), wrap it "
    "in parentheses, e.g. \\boxed{(2, 3), 7}."
)
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Read the problem and the answer choices, "
    "then select the single best answer. After your reasoning, output ONLY the "
    "letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}. "
    "The very last thing in your response must be that \\boxed{<letter>}."
)


def build_chat(question, options):
    """Build the chat messages for one question (MATH vs MCQ system prompt)."""
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts = "\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))
        return [{"role": "system", "content": SYSTEM_PROMPT_MCQ},
                {"role": "user", "content": f"{question}\n\nOptions:\n{opts}"}]
    return [{"role": "system", "content": SYSTEM_PROMPT_MATH},
            {"role": "user", "content": question}]


def load_jsonl(path):
    """Read a JSONL file into a list of dicts (inlined from harness.load_jsonl)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Self-consistency generation (eddie/self_consistency.ipynb §6–§7)
# ──────────────────────────────────────────────────────────────────────────────

def generate_samples(private_path, samples_path):
    """Sample N_SAMPLES completions per private question and archive every raw
    sample to `samples_path` (append-only, resumable). Returns `samples_path`.

    The archive record per question is:
        {'id', 'model_tag', 'is_mc', 'n_samples', 'temperature',
         'max_gen_tokens', 'samples': [{'text', 'truncated', 'n_tok'}, ...]}
    which is exactly the format Stage 2 (post-processing) expects.
    """
    # vLLM prefers the 'spawn' multiproc method on Colab/Linux.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # Lazy imports: heavy GPU deps are only needed for generation, so the module
    # stays importable (e.g. for Stage 2 / on a machine without vLLM).
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if not os.path.exists(private_path):
        raise FileNotFoundError(
            f"private set not found at {private_path}. Pass private_path=... or "
            f"place the file there."
        )
    rows = load_jsonl(private_path)

    print(f"[stage1] loading model {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    llm = LLM(model=MODEL_ID, dtype="bfloat16", trust_remote_code=True,
              max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=GPU_MEM_UTIL,
              seed=SEED, enforce_eager=True)
    sp = SamplingParams(n=N_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P,
                        top_k=TOP_K, min_p=0.0, max_tokens=MAX_GEN_TOKENS, seed=SEED)
    print("[stage1] model loaded")

    # Resume from the archive: a problem is 'done' once its samples are written.
    done = set()
    if os.path.exists(samples_path):
        with open(samples_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])
    todo = [r for r in rows if r["id"] not in done]
    print(f"[stage1] total={len(rows)} archived={len(done)} todo={len(todo)}")

    with open(samples_path, "a", encoding="utf-8") as fout:
        for ci in range(0, len(todo), CHUNK):
            chunk = todo[ci:ci + CHUNK]
            prompts = [tok.apply_chat_template(build_chat(r["question"], r.get("options")),
                                               tokenize=False, add_generation_prompt=True)
                       for r in chunk]
            outs = llm.generate(prompts, sp)
            for r, out in zip(chunk, outs):
                rec = {
                    "id": r["id"],
                    "model_tag": MODEL_TAG,
                    "is_mc": bool(r.get("options")),
                    "n_samples": len(out.outputs),
                    "temperature": TEMPERATURE,
                    "max_gen_tokens": MAX_GEN_TOKENS,
                    "samples": [{"text": o.text,
                                 "truncated": (o.finish_reason == "length"),
                                 "n_tok": len(o.token_ids)} for o in out.outputs],
                }
                fout.write(json.dumps(rec) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            print(f"  [stage1] {min(ci + CHUNK, len(todo))}/{len(todo)} archived")

    print(f"[stage1] raw samples -> {samples_path}")
    return samples_path


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2 — Post-processing v2 aggregation (eddie/Post_processing.ipynb)
# ──────────────────────────────────────────────────────────────────────────────

def _get_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        ss = [v for v in x.values() if isinstance(v, str)]
        return max(ss, key=len) if ss else ""
    return str(x)


def _find_all_boxes(s):
    out, i = [], 0
    while True:
        j = s.find("\\boxed{", i)
        if j < 0:
            break
        k = j + 7
        d = 1
        while k < len(s) and d:
            d += (s[k] == "{") - (s[k] == "}")
            k += 1
        out.append(s[j + 7:k - 1])
        i = k
    return out


def _split_top(s):                       # top-level comma split, NON-destructive (keeps raw text)
    out, depth, start = [], 0, 0
    for i, ch in enumerate(s):
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        elif ch == "," and depth <= 0:
            out.append(s[start:i].strip())
            start = i + 1
    out.append(s[start:].strip())
    return [x for x in out if x]


def _sub_answers(text):                   # RAW sub-answers from boxes; None if no box (drops fallback)
    seg = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    bx = _find_all_boxes(seg) or _find_all_boxes(text)
    if not bx:
        return None
    items = []
    for b in bx:
        items += _split_top(b)
    return items or None


def _is_complete(text):
    te = text.rfind("</think>")
    return te >= 0 and "\\boxed{" in text[te + 8:]


def aggregate_submission(samples_path, private_path, out_csv):
    """Post-processing v2: per-[ANS]-slot vote over the archived raw samples,
    writing the final submission CSV. Returns `out_csv`.
    """
    from judger import Judger          # official grader from eddie/
    J = Judger(strict_extract=False)

    # [ANS] counts per question id (the number of blanks to fill / slots to vote).
    qcount = {}
    for line in open(private_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)
        qcount[str(q["id"])] = q.get("question", "").count("[ANS]")

    def nz(v):
        try:
            return J.norm_ans_str(v)
        except Exception:
            return v

    def vote_slot(raw_vals):              # vote on normalized key, RETURN a raw string
        groups = {}
        for v in raw_vals:
            groups.setdefault(nz(v), []).append(v)
        keys = list(groups)
        parent = {k: k for k in keys}

        def find(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                if find(keys[a]) != find(keys[b]):
                    try:
                        eq = J.is_equal(keys[a], keys[b])
                    except Exception:
                        eq = (keys[a] == keys[b])
                    if eq:
                        parent[find(keys[a])] = find(keys[b])
        merged = {}
        for k in keys:
            merged.setdefault(find(k), []).append(k)
        best = max(merged.values(), key=lambda g: sum(len(groups[k]) for k in g))
        raws = [v for k in best for v in groups[k]]
        return Counter(raws).most_common(1)[0][0]     # raw, real string that appeared

    def fmt(s):
        s = s.strip()
        return (f"({s})" if len(_split_top(s)) > 1
                and not (s.startswith("(") and s.endswith(")")) else s)

    final = {}
    for line in open(samples_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        qid = str(r["id"])
        is_mc = bool(r.get("is_mc"))
        cand = []                                      # (text, raw_sub_answers, complete)
        for t in map(_get_text, r["samples"]):
            if not t:
                continue
            sa = _sub_answers(t)
            if sa is None:
                continue                               # no box -> doesn't vote
            cand.append((t, sa, _is_complete(t)))
        if not cand:
            final[qid] = next((_get_text(s) for s in r["samples"] if _get_text(s)), "")
            continue

        if is_mc:
            K = 1
        else:
            K = qcount.get(qid, 0)
            if K <= 0:
                c = Counter(len(sa) for _, sa, _ in cand)
                K = c.most_common(1)[0][0]
        well = [p for p in cand if len(p[1]) == K]
        if not well:
            c = Counter(len(sa) for _, sa, _ in cand)
            K = c.most_common(1)[0][0]
            well = [p for p in cand if len(p[1]) == K]

        voted = [vote_slot([sa[i] for _, sa, _ in well]) for i in range(K)]
        combined = voted[0].strip().upper() if is_mc else ", ".join(fmt(s) for s in voted)
        vkey = tuple(nz(v) for v in voted)
        rep = max(well, key=lambda p: (p[2], sum(nz(a) == b for a, b in zip(p[1], vkey))))
        final[qid] = rep[0].rstrip() + "\n\nThe final answer is \\boxed{" + combined + "}"

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["id", "response"])
        for qid in sorted(final, key=lambda s: (0, int(s)) if s.isdigit() else (1, s)):
            w.writerow([qid, final[qid]])
    print(f"[stage2] wrote {out_csv} | {len(final)} questions")
    return out_csv


# ──────────────────────────────────────────────────────────────────────────────
# Single entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(private_path=DEFAULT_PRIVATE_PATH,
                  output_dir=DEFAULT_OUTPUT_DIR,
                  skip_generation=False):
    """Run the full pipeline end-to-end and write the submission CSV.

    Args:
        private_path: path to the private-set JSONL (same schema as data/public.jsonl,
            with a `question` field, optional `options`, and an `id`).
        output_dir: directory for all outputs (created if missing). Defaults to
            ./outputs/tester so existing notebook artifacts are never overwritten.
        skip_generation: if True, skip Stage 1 and aggregate from an existing
            raw-sample archive in `output_dir` (useful to re-run only Stage 2).

    Returns:
        Path to the final submission CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples_path = os.path.join(output_dir, f"{MODEL_TAG}__private__samples.jsonl")
    out_csv      = os.path.join(output_dir, f"{MODEL_TAG}_submission.csv")

    if not skip_generation:
        generate_samples(private_path, samples_path)         # Stage 1 (GPU, ~20h on A100-80GB)
    elif not os.path.exists(samples_path):
        raise FileNotFoundError(
            f"skip_generation=True but no archive at {samples_path}"
        )

    aggregate_submission(samples_path, private_path, out_csv)  # Stage 2 (CPU, minutes)
    print(f"[done] submission -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CSE 151B end-to-end inference -> submission CSV")
    p.add_argument("--private-path", default=DEFAULT_PRIVATE_PATH,
                   help="path to the private-set JSONL (default: data/private.jsonl)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="output directory (default: outputs/tester)")
    p.add_argument("--skip-generation", action="store_true",
                   help="skip Stage 1; aggregate from an existing raw-sample archive")
    args = p.parse_args()
    run_inference(private_path=args.private_path,
                  output_dir=args.output_dir,
                  skip_generation=args.skip_generation)
