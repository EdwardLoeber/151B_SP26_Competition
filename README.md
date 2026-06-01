# CSE 151B Competition — Final Submission

Single end-to-end pipeline that reproduces our Kaggle submission. Calling
**`run_inference()`** loads the model, runs inference on the private set, applies
all post-processing, and writes the final submission CSV — nothing manual.

## Method (one line)

Self-consistency over the **untouched base model** `Qwen/Qwen3-4B-Thinking-2507`:
sample N=7 completions per question, then vote each answer slot with
judger-equivalence clustering. We did **not** fine-tune for the final submission
(GRPO came in at 70.41% vs the 71.60% base), so there are **no custom weights to
download** — the base model loads straight from the Hugging Face Hub.

## Hardware & inference time

| | |
|---|---|
| **GPU** | 1× NVIDIA **A100 80GB** (Google Colab) |
| **Stage 1 — generation** | **~20 hours** for the full private set (N=7, 32,768-token thinking budget) |
| **Stage 2 — aggregation** | a few minutes (CPU only) |

An H100 is roughly 1.6× faster on Stage 1.

## Model weights

Nothing to download or place anywhere. We did not fine-tune, so requirement #3
(uploading a fine-tuned checkpoint to the HF Hub) does **not** apply. The base
model is pulled from the Hub by ID on first run and cached normally:

```
Qwen/Qwen3-4B-Thinking-2507
```

## Setup

Python 3.10+ with a CUDA 12.6 GPU. Install the same stack used to produce the
submission:

```bash
pip install uv
uv pip install --system torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
uv pip install --system "vllm==0.9.2" "transformers==4.53.3" sympy "antlr4-python3-runtime==4.11.1"
```

Place the private set at `data/private.jsonl` (same schema as `data/public.jsonl`:
a `question` field, optional `options` for multiple-choice, and an `id`), or pass
its path explicitly (see below).

## How to reproduce — `run_inference()`

```python
from run_inference import run_inference

# Full pipeline: generation (Stage 1) + post-processing (Stage 2).
run_inference()                                   # uses data/private.jsonl
run_inference(private_path="/path/to/private.jsonl")
```

Or from the command line:

```bash
python run_inference.py --private-path data/private.jsonl
```

Outputs are written under **`outputs/tester/`**:

| File | Contents |
|---|---|
| `tester__private__samples.jsonl` | raw per-question archive (all 7 samples each) — Stage 1 output, resumable |
| `tester_submission.csv` | **final submission** (`id`, `response`) — Stage 2 output |

Stage 1 is resumable: if it's interrupted, re-running skips questions already in
the archive. To re-run only the fast aggregation against an existing archive:

```python
run_inference(skip_generation=True)
```

## What the pipeline does (inside `run_inference()`)

1. **Stage 1 — generation** (from `eddie/self_consistency.ipynb`): loads
   `Qwen/Qwen3-4B-Thinking-2507` with vLLM and samples **N=7** completions per
   question (temp 0.6, top_p 0.95, top_k 20, seed 151, 32,768-token budget),
   archiving every raw sample.
2. **Stage 2 — post-processing** (from `eddie/Post_processing.ipynb`): for each
   question, parses the `\boxed{}` sub-answers of every sample, votes each `[ANS]`
   slot independently using the official judger's numeric/symbolic equivalence to
   cluster votes, rebuilds the final answer, and writes the CSV.

All hyperparameters in `run_inference.py` are the final ones used for the
submission (no external configuration needed).

## Repo layout

| Path | Description |
|---|---|
| `run_inference.py` | **Single entry point** — full pipeline end-to-end |
| `eddie/self_consistency.ipynb` | Stage 1 source notebook (generation) |
| `eddie/Post_processing.ipynb` | Stage 2 source notebook (aggregation) |
| `eddie/judger.py`, `eddie/utils.py` | Official answer grader (imported by `run_inference.py`) |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `outputs/tester/` | Created at runtime; holds the archive and final CSV |
