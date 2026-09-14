# 2. Evaluations

**What evaluation is trying to do:** measure *functional* correctness, not text
similarity — did the model call the right function, with the right parameters, and
*only* when a function call was actually warranted? Three failure modes matter
independently and are tracked separately rather than blended into one accuracy number:
wrong/hallucinated function calls, wrong parameters on an otherwise-correct call, and
missed refusals (calling something when it should have asked for clarification or
declined).

**How it's applied to the trained model:** each checkpoint from `1_training/` is served
locally (e.g. via vLLM's OpenAI-compatible server) and run through the suites below
before it's allowed to be promoted; internal, risk-tiered results are the actual gate —
BFCL is a supporting/comparability signal, not the other way around.

**Baseline model: `Qwen/Qwen3-8B`** — the same base checkpoint `1_training/1_sft/train_sft.py`
fine-tunes (see its README for why that consistency matters). Qwen3 is a hybrid
reasoning model: it can emit `<think>...</think>` before answering. `run_internal_eval.py`
and `run_latency_benchmark.py` expose `--enable-thinking` (default **off**) via the chat
template, matching production (which can't afford chain-of-thought latency).
`run_bfcl_eval.py` is the one exception, not by choice: BFCL's own OSS "Prompt"-style
handler builds prompts by hand and calls the raw `/v1/completions` endpoint, never
`/v1/chat/completions` — no chat template runs in that path at all, so there's no
`enable_thinking` to set (verified against the installed `bfcl-eval` package's handler
code, not assumed).

## Code structure

`run_bfcl_eval.py` implements this project's `Evaluator` interface —
`load_eval_dataset() -> run_predictions(eval_dataset) -> compute_metrics(predictions)` —
with `BFCLEvaluator` as the BFCL implementation, built on the real `bfcl-eval` CLI
(package `bfcl-eval`, command `bfcl`; verified against the installed package, not
assumed — see the class docstring):

- **`load_eval_dataset()`** — resolves `test_category` into BFCL's own concrete test
  names and backing files (`bfcl_eval.constants.category_mapping`), so a typo'd category
  fails fast instead of burning GPU time. BFCL owns its eval data internally; this
  project doesn't supply it.
- **`run_predictions(eval_dataset)`** — `bfcl generate`. By default BFCL self-hosts the
  model (launches its own vLLM/SGLang server, downloads weights from the Hub if needed,
  generates, tears the server down) — pass `local_model_path` for a local checkpoint
  (e.g. a merged fine-tuned model), or `skip_server_setup=True` to point at an
  already-running server instead.
- **`compute_metrics(predictions)`** — `bfcl evaluate` (scores the generated results)
  then `bfcl scores` (prints the leaderboard).

`BFCLEvaluator.__init__` validates `model` against BFCL's own `MODEL_CONFIG_MAPPING` and
warns if you pass a `-FC` id — this project's fine-tune uses prompting-style tool calls
(tools as text in the system message, calls parsed from generated text), matching BFCL's
non-`-FC` ("Prompt") registrations, not the "-FC" ones (which drive the model's native
tool-calling API format).

## Bench scope table

| Suite | Script | What it measures | Gate |
|---|---|---|---|
| BFCL (`python` category group) | `run_bfcl_eval.py` | function-call correctness on BFCL's own Python-relevant categories — simple, parallel/multiple, irrelevance/relevance, live variants | regression check vs. similarly-sized open baselines |
| Internal – accuracy | `run_internal_eval.py` | function-name + parameter correctness on the real internal API catalog, by risk tier | ≥ 98% name accuracy, ≥ 99% on high-risk tier |
| Internal – hallucination | `run_internal_eval.py` | calls to functions that don't exist / invalid params | < 0.5% overall, 0% on high-risk tier |
| Internal – refusal | `run_internal_eval.py` | correctly declines/asks for clarification instead of guessing | ≥ 98% |
| Latency under load | `run_latency_benchmark.py` | p50/p95 latency at production concurrency | vs. per-workflow SLA |

Internal rows are broken out by `risk_tier` (currently all `unclassified` — see the
`0_data/` README's placeholder note — until real internal-API examples replace it).

## Accuracy by pipeline stage

Each stage is run through the bench-scope suites above; this is what shows fine-tuning
earns its keep over the incumbent model, and that quantization (`3_optimizations/`)
doesn't quietly cost accuracy.

| Stage | BFCL accuracy | Internal accuracy | Hallucination rate | Refusal accuracy |
|---|---|---|---|---|
| Incumbent (Qwen3-8B, zero-shot, no fine-tune, non-thinking) | *measured* | *measured* | *measured* | *measured* |
| Post-SFT (LoRA, bf16) | *measured* | *measured* | *measured* | *measured* |
| Post-SFT + quantized (FP8/AWQ) | *measured* | *measured* | *measured* | *measured* |

## Latency by pipeline stage

Every row is captured at the **32-concurrent production load** specifically — the load
level the fraud-detection SLA actually has to hold under, not idle/single-stream numbers.

| Stage | Median tok/s | TTFT (p50) | Inter-token latency (p50) |
|---|---|---|---|
| Incumbent (bf16, non-thinking) | *measured* | *measured* | *measured* |
| Post-SFT (bf16) | *measured* | *measured* | *measured* |
| Post-SFT + FP8/AWQ | *measured* | *measured* | *measured* |

## Scripts

### `run_bfcl_eval.py`

```bash
python run_bfcl_eval.py --model Qwen/Qwen3-8B --test-category python --num-gpus 1
```

To evaluate a fine-tuned checkpoint instead: merge the LoRA adapter into the base
weights (see `1_training/README.md`) and pass `--local-model-path` pointing at the
merged directory, keeping `--model Qwen/Qwen3-8B` (same architecture/tokenizer/handler).
This isn't executable in a GPU-less environment — verified here against the installed
`bfcl-eval` package's real CLI (`bfcl models` / `bfcl test-categories` /
`--help` on each subcommand) and the non-GPU parts of `BFCLEvaluator`
(`load_eval_dataset`, model/category validation), not by actually running generation.

### `run_internal_eval.py`

Runs `0_data/data/test.jsonl` through a locally served checkpoint (OpenAI-compatible
endpoint), parses the model's output as a function call, and scores name accuracy,
parameter accuracy, hallucination rate, and refusal accuracy — broken out by risk tier.

```bash
python run_internal_eval.py --endpoint http://localhost:8000/v1 --model Qwen/Qwen3-8B \
  --test-file ../0_data/data/test.jsonl --output results/internal_eval.json
```

### `run_latency_benchmark.py`

A concurrency-swept load test (1, 8, 16, 24, 32 concurrent requests) against the same
local endpoint, streaming completions to measure TTFT and inter-token latency directly,
and reporting p50/p90/p99 alongside aggregate throughput. Shares its concurrency sweep
with `3_optimizations/benchmark_serving.py`.

```bash
python run_latency_benchmark.py --endpoint http://localhost:8000/v1 --model Qwen/Qwen3-8B \
  --prompts-file ../0_data/data/test.jsonl --concurrency 1 8 16 24 32
```

### `log_vllm_bench_to_mlflow.py`

Runs `vllm`'s own built-in `vllm bench serve` (real tool-calling traffic via its
`BFCLDataset` loader — see root README's step 5) as a subprocess and logs its
mean/median/p50/p95/p99 TTFT, TPOT, inter-token latency, and request/output/total-token
throughput to MLflow, under the `fin-agent-vllm-bench` experiment. Always passes
`--metric-percentiles 50,95,99` (vLLM's own default is p99 only).

```bash
python log_vllm_bench_to_mlflow.py \
  --mlflow-tracking-uri http://localhost:5000 \
  --base-url http://localhost:8000 --model Qwen/Qwen3-8B \
  --bfcl-categories simple,multiple,parallel,parallel_multiple --concurrencies 16,32,64
```

`--max-concurrency <n>` runs a single level, logged as its own run with plain
(unprefixed) metric/param names. `--concurrencies` (comma-separated) sweeps several
levels — runs the full benchmark once per value but logs them all into **one** MLflow
run, each value's fields prefixed `c<N>_` (e.g. `c32_mean_ttft_ms`) so they sit side by
side instead of colliding. The prefix isn't cosmetic: MLflow params are immutable per
key, so logging the same unprefixed key twice with a different value (e.g. `date`,
which differs every `vllm bench serve` invocation) raises `INVALID_PARAMETER_VALUE` on
the second concurrency — the same class of bug `log_bfcl_to_mlflow.py` already works
around for multi-category runs. Pass exactly one of `--max-concurrency` /
`--concurrencies`.

Bare-host analogue: `tox -e vllm-bench -- <same flags>` (see `tox.ini`'s
`[testenv:vllm-bench]`). Requires `kubectl -n fin-agent port-forward svc/mlflow
5000:5000` running, same as this file's "Evaluating a specific MLflow run against BFCL"
section below — and, separately, a live `kubectl -n fin-agent port-forward svc/
fin-agent-vllm-<name> 8000:8000` for the vLLM endpoint itself if you're serving via
`../configs/templates/inference/vllm-serve-checkpoint.yaml`. Both port-forwards have to
stay running in their own terminals for the whole benchmark; a dead one just produces a
flat connection-refused error on every request rather than a useful one. Prefer not to
juggle either: `../configs/templates/inference/vllm-bench-mlflow-checkpoint-job.yaml`
runs the serve + benchmark + MLflow-logging entirely in-cluster (see root README's step
5, Option A) — no port-forwarding at all.

## Evaluating a specific MLflow run against BFCL

Every checkpoint this project trains gets logged to MLflow as a `model` artifact
(`train_sft.py`/`train_grpo.py` with `--mlflow` — see `1_training/`), identified by a
`run_id`. This section is about pinning a BFCL evaluation to one specific tracked run,
rather than whatever checkpoint happens to be sitting on local/hostPath disk right now —
useful for reproducing a specific result later, or comparing two runs side by side.

There are two ways to run this: entirely inside the cluster (kube Jobs), or on a bare GPU
host with `tox` and no kube involvement beyond reaching MLflow itself. Pick based on where
your GPU actually is.

### Option A — entirely in-cluster

`configs/templates/inference/run-bfcl-eval-run-ids-suite.sh` evaluates the raw baseline
plus up to two MLflow run_ids (SFT and GRPO), each served via its own vLLM Deployment
that downloads the checkpoint by `run_id`
(`configs/templates/inference/bfcl-eval-mlflow-checkpoint-job.yaml`), runs BFCL, logs
`bfcl_non_live_ast_accuracy`/`bfcl_live_ast_accuracy` to MLflow, then tears the
Deployment down (the generated results themselves are *not* deleted — see the script's
own comments for why that matters if a later run needs to recover from a logging-only
bug without regenerating everything).

```bash
MODELS="sft" SFT_RUN_ID=7c63856e88a54b7c907368b04350edb4 \
  ./configs/templates/inference/run-bfcl-eval-run-ids-suite.sh
```

Defaults to a cheap smoke pass (`parallel`+`live_parallel`, a couple hundred cases).
Add `FULL_SCALE=1` for the real, leaderboard-comparable `python` category (hours, not
minutes). `MODELS` also accepts `baseline` and `grpo`; `GRPO_RUN_ID` overrides that leg's
run_id the same way `SFT_RUN_ID` does.

**Just want an endpoint for that run_id, no BFCL eval?** Reuse the same Deployment+Service
on their own (filters the Job out of the rendered manifest):

```bash
python3 -c "
src=open('configs/templates/inference/bfcl-eval-mlflow-checkpoint-job.yaml').read()
src=src.replace('__NAME__','sft').replace('__RUN_ID__','7c63856e88a54b7c907368b04350edb4')
print('\n---\n'.join(d for d in src.split('\n---\n') if 'kind: Job' not in d))
" | kubectl apply -f -
kubectl -n fin-agent rollout status deploy/fin-agent-vllm-sft --timeout=900s
```

Endpoint: `fin-agent-vllm-sft.fin-agent.svc.cluster.local:8000` (in-cluster only — see
Option B below to reach it from outside). Tear down when done:

```bash
kubectl -n fin-agent delete deployment fin-agent-vllm-sft
kubectl -n fin-agent delete service fin-agent-vllm-sft --ignore-not-found
```

For speculative decoding, use `configs/templates/inference/vllm-serve-checkpoint-fp8-dflash.yaml`
(DFlash, not MTP) — MTP needs a checkpoint with a trained MTP head, which every checkpoint
this project actually produces (a plain dense Qwen3-8B merge) doesn't have; a
`--speculative-config '{"method": "mtp", ...}'` config fails to start on one. DFlash pairs
a separate, standalone drafter model (`z-lab/Qwen3-8B-DFlash-b16`) with the target via
rejection sampling instead, so it works against this project's own fine-tunes — see that
file's header for the full explanation and the verified vLLM config.

### Option B — bare GPU host, no kube (via `tox`)

`tox.ini` has two envs for this, both bare-`pip`-installed (not the NGC/vLLM container
images the training envs assume — see `tox.ini`'s own header for why `sitepackages` is
off for both):

- **`bfcl`** — runs `log_bfcl_to_mlflow.py`. By default it self-hosts vLLM itself (`bfcl
  generate` shells out to `vllm serve <model> ...` as a subprocess and tears it down when
  finished) — no separate serve step for a one-shot eval.
- **`vllm-serve`** — runs `serve_mlflow_checkpoint.py`: downloads one run's checkpoint and
  execs into a **long-running** `vllm serve`. Use this when you want a standing endpoint
  to reuse across several evals, or to poke at directly (curl, `run_internal_eval.py`,
  etc.), rather than reloading the model fresh every time.

MLflow itself is still in-cluster either way — both envs' baked-in tracking URI is
`http://localhost:5000`, which needs a port-forward running in its own terminal for the
whole session:

```bash
kubectl -n fin-agent port-forward svc/mlflow 5000:5000
```

**Self-hosted, one-shot** (simplest — no separate serve step, no extra port-forward):

```bash
tox -e bfcl -- --model Qwen/Qwen3-8B --test-category python \
  --local-model-path ~/.cache/fin-agent/mlflow-models/7c63856e88a54b7c907368b04350edb4/model
```

`--local-model-path` needs the checkpoint already downloaded — either it's already cached
there from a prior `vllm-serve` run (below), or point `--local-model-path` anywhere else
you've already got the checkpoint on disk.

**Standing server + separate eval** (reusable across several eval runs without reloading
the model each time):

```bash
# terminal 1 -- host the model, stays running
kubectl -n fin-agent port-forward svc/mlflow 5000:5000
```
```bash
# terminal 2 -- serve it
tox -e vllm-serve -- --run-id 7c63856e88a54b7c907368b04350edb4
```
```bash
# terminal 3 -- run the eval against the server from terminal 2 (same host, so
# plain localhost:8000 -- no port-forward needed for vLLM itself, only for MLflow)
tox -e bfcl -- --model Qwen/Qwen3-8B --test-category python --skip-server-setup
```

`--skip-server-setup` is the flag that matters here — without it, `bfcl` tries to launch
its *own* `vllm serve` instead of using the one already running from terminal 2.

**Two things worth knowing before running either for the first time on a new host:**

- `serve_mlflow_checkpoint.py` (used by `vllm-serve`, and needed once to populate the
  local cache `--local-model-path` above reads from) has to work around this project's
  MLflow being configured with a bare local-path artifact store (see
  `configs/setup/mlflow.yaml`'s header) — resolvable by the normal MLflow client only from
  *inside* the cluster. It falls back to reading directly from the PVC's real hostPath
  (`/var/lib/fin-agent/mlflow-artifacts`) when that happens; see the script's own
  `INCLUSTER_ARTIFACT_ROOT` comment. That fallback needs your host user to actually have
  read access to that path — a `Permission denied` there needs `sudo usermod`/`chmod` on
  the host, not a code fix.
- `vllm`'s first real request compiles a small CUDA kernel via a system C compiler, which
  needs Python's development headers (`Python.h`) — not something `tox`/`pip` installs.
  `fatal error: Python.h: No such file or directory` means `sudo apt-get install
  python3.12-dev` (match your actual Python minor version). If you can't install the
  headers right now, `tox -e vllm-serve -- --run-id ... --enforce-eager` sidesteps the
  compile step entirely (slower inference, no compiler needed) — that flag passes
  straight through to `vllm serve` (`serve_mlflow_checkpoint.py` forwards anything it
  doesn't recognize). `tox -e bfcl`'s self-hosted mode can't do the same: bfcl-eval's own
  subprocess call that launches `vllm serve` uses a fixed argument list with no
  passthrough (verified against the installed package's source, not assumed) — if you hit
  this on the `bfcl` env specifically, install the headers, or serve via `vllm-serve`
  first and point `bfcl` at it with `--skip-server-setup` instead.
