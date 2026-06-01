"""
Phase 0 — Local scoring harness.

This wraps the EXACT scoring path used by the competition's starter notebook:
  - MC questions  -> score_mcq() : letter regex, judger.py is NOT involved.
  - free-form     -> Judger.auto_judge(pred, gold_list, options=[[]]*k).

We do NOT reimplement grading. We import the official judger.py / utils.py and
replicate the starter's wiring verbatim. The only things added here are:
  - bucketed accuracy (mc / free_single / free_multi)
  - diagnostics (extraction failures, MC-fallback firings, truncation flags)

A "prediction" is a dict {"id": int, "response": str}. A "row" is a line from
public.jsonl: {"id", "question", "answer", optional "options"}.

Verified facts about the grader (see test_harness.py for executable proof):
  - MC is graded by letter ONLY. The gold for ~10 "MC" rows is a numeric value,
    not a letter; those are effectively ungradable as MC. That is the grader's
    behavior, not ours.
  - The MC fallback regex r"\\b([A-Z])\\b"[-1] grabs the last standalone capital
    ANYWHERE in the trace if no \\boxed{<letter>} is present. This can mis-grade.
    score_is_mc_fallback() flags when a correct MC score came only via fallback.
  - free-form uses relative-error <= ~1e-8. Rounded decimals FAIL; exact
    fractions / symbolic forms / >=10-sig-fig decimals pass.
"""

import re
import sys
import json
from pathlib import Path

# Import the OFFICIAL grader from the same directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judger import Judger  # noqa: E402


# ── MC scoring: copied verbatim from starter Cell 22 ─────────────────────────

def extract_letter(text: str) -> str:
    """Starter's MC extractor. Primary: \\boxed{<one letter>}. Fallback: last
    standalone capital letter in the (uppercased) text."""
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-Z])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    return extract_letter(response) == str(gold_letter).strip().upper()


def mc_via_fallback(response: str) -> bool:
    """True if extract_letter would rely on the fallback (no boxed single
    letter present). Used to flag fragile MC scores."""
    return re.search(r"\\boxed\{([A-Za-z])\}", response) is None


# ── Bucketing ────────────────────────────────────────────────────────────────

def bucket_of(row: dict) -> str:
    if row.get("options"):
        return "mc"
    ans = row["answer"]
    k = len(ans) if isinstance(ans, list) else 1
    return "free_single" if k == 1 else "free_multi"


# ── Core scoring of one prediction against one row ───────────────────────────

import signal as _signal


class _GraderTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _GraderTimeout()


def score_one(row: dict, response: str, judger: Judger, timeout: int = 0) -> dict:
    """Return a per-row result dict. Mirrors starter Cell 22 exactly.

    timeout > 0 wraps the (possibly slow) sympy grading in a SIGALRM timeout.
    On hang/error the row is graded INCORRECT (never skipped) so the denominator
    stays complete — a row the grader can't score in time is a row you lose.
    NOTE: signal-based timeout only works on the main thread; leave timeout=0
    when calling from worker threads (e.g. concurrent distillation)."""
    is_mc = bool(row.get("options"))
    gold = row["answer"]

    diag = {
        "id": row["id"],
        "bucket": bucket_of(row),
        "is_mc": is_mc,
        "extract_fail": False,   # free-form: grader extracted empty answer
        "mc_fallback": False,    # mc: scored without a boxed letter
        "truncated": False,      # set by caller if generation hit token cap
        "timed_out": False,      # grading exceeded the timeout -> counted wrong
    }

    if timeout > 0:
        _signal.signal(_signal.SIGALRM, _alarm_handler)
        _signal.alarm(timeout)
    try:
        if is_mc:
            correct = score_mcq(response, str(gold))
            diag["mc_fallback"] = mc_via_fallback(response)
        else:
            gold_list = gold if isinstance(gold, list) else [gold]
            try:
                extracted = judger.extract_ans(response)
                diag["extract_fail"] = (not extracted)
            except _GraderTimeout:
                raise                      # let the outer handler flag the timeout
            except Exception:
                diag["extract_fail"] = True
            try:
                correct = judger.auto_judge(
                    pred=response,
                    gold=gold_list,
                    options=[[]] * len(gold_list),
                )
            except _GraderTimeout:
                raise                      # let the outer handler flag the timeout
            except Exception:
                correct = False
    except _GraderTimeout:
        diag["timed_out"] = True
        correct = False          # a row we can't grade in time is a LOSS, not a skip
    finally:
        if timeout > 0:
            _signal.alarm(0)

    diag["correct"] = bool(correct)
    return diag


# ── Batch scoring ────────────────────────────────────────────────────────────

def load_jsonl(path) -> list:
    return [json.loads(line) for line in open(path) if line.strip()]


def score(predictions, rows, truncated_ids=None, strict_extract=False, timeout=2):
    """
    predictions : list of {"id": int, "response": str}  OR  dict id->response
    rows        : list of public.jsonl row dicts (must contain the ids in preds)
    truncated_ids : optional set of ids whose generation hit the token cap
    timeout     : per-row grading timeout in seconds (SIGALRM). A row that can't
                  be graded in time is counted INCORRECT, never skipped, so the
                  denominator is always complete. Runs single-threaded on the
                  main thread, so the signal timeout is valid here.
    returns     : (summary dict, per_row list of diag dicts)

    Fails loud: if any prediction id is missing from rows (or vice-versa for the
    ids being scored), raises rather than silently skipping.
    """
    judger = Judger(strict_extract=strict_extract)
    truncated_ids = set(truncated_ids or [])

    if isinstance(predictions, dict):
        pred_map = {int(k): v for k, v in predictions.items()}
    else:
        pred_map = {int(p["id"]): p["response"] for p in predictions}

    row_map = {int(r["id"]): r for r in rows}

    missing = [pid for pid in pred_map if pid not in row_map]
    if missing:
        raise ValueError(f"{len(missing)} prediction ids absent from rows, "
                         f"e.g. {missing[:5]}")

    per_row = []
    for pid, resp in pred_map.items():
        row = row_map[pid]
        d = score_one(row, resp, judger, timeout=timeout)
        if pid in truncated_ids:
            d["truncated"] = True
        per_row.append(d)

    summary = summarize(per_row)
    return summary, per_row


def summarize(per_row) -> dict:
    buckets = ["mc", "free_single", "free_multi"]
    out = {}
    for b in buckets + ["overall"]:
        subset = per_row if b == "overall" else [d for d in per_row if d["bucket"] == b]
        n = len(subset)
        c = sum(d["correct"] for d in subset)
        out[b] = {
            "n": n,
            "correct": c,
            "acc": (c / n) if n else 0.0,
        }
    # diagnostics
    out["diagnostics"] = {
        "extract_fail": sum(d["extract_fail"] for d in per_row),
        "mc_fallback_total": sum(d["mc_fallback"] for d in per_row),
        "mc_fallback_and_correct": sum(d["mc_fallback"] and d["correct"]
                                       for d in per_row),
        "truncated": sum(d["truncated"] for d in per_row),
        "timed_out": sum(d.get("timed_out", False) for d in per_row),
    }
    return out


def print_summary(summary: dict):
    print("=" * 56)
    print("EVALUATION RESULTS")
    print("=" * 56)
    for b, label in [("mc", "MC         "),
                     ("free_single", "Free-1blank"),
                     ("free_multi", "Free-multi "),
                     ("overall", "Overall    ")]:
        s = summary[b]
        print(f"  {label}: {s['correct']:4d} / {s['n']:4d}  ({s['acc']*100:6.2f}%)")
    d = summary["diagnostics"]
    print("-" * 56)
    print(f"  extract_fail (free-form, empty extract): {d['extract_fail']}")
    print(f"  MC scored via fallback regex          : {d['mc_fallback_total']}")
    print(f"    ...of which graded CORRECT (fragile): {d['mc_fallback_and_correct']}")
    print(f"  generations truncated at token cap    : {d['truncated']}")
    print(f"  grading timed out (counted INCORRECT)  : {d.get('timed_out', 0)}")
    print("=" * 56)
