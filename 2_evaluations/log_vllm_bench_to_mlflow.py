"""Run `vllm bench serve` and log its scalar metrics (TTFT, TPOT, ITL, throughput, ...)
to MLflow -- vllm bench serve itself has no MLflow integration, only its own local
--save-result JSON file. This bridges that gap the same way log_bfcl_to_mlflow.py bridges
it for bfcl-eval: run the real tool as a subprocess, point --save-result at a temp file,
and log every scalar key from the result JSON as an MLflow metric or param.

Key names logged as metrics -- verified against vLLM's own benchmarks/serve.py source,
not assumed -- include mean/median/std/p50/p95/p99 for ttft_ms, tpot_ms, itl_ms, e2el_ms
(this script always passes --metric-percentiles 50,95,99 -- vllm bench serve's own
default is p99 only), plus request_throughput, output_throughput,
total_token_throughput, completed, failed. Per-request raw lists (ttfts, itls, latencies,
generated_texts, errors, ...) are skipped -- MLflow metrics must be scalars, and these
are exactly the detail save_to_pytorch_benchmark_format's own ignored_metrics list
already excludes for the same reason.

Also logs `kv_cache_usage_perc_mean`/`_max`: `vllm bench serve` only ever sees
client-observed latency/throughput, never the server's own KV-cache occupancy, so this
script separately polls the target server's `/metrics` endpoint (Prometheus text format,
`vllm:kv_cache_usage_perc` -- exposed by vLLM's OpenAI-compatible server by default, no
extra server flag needed) once a second for the duration of each `vllm bench serve` call.
Missing if the endpoint is unreachable or the vLLM build doesn't expose it -- logged as a
stderr warning, not a failure.

Usage (against an already-running, OpenAI-compatible vLLM server -- e.g.
configs/templates/inference/vllm-serve-checkpoint.yaml port-forwarded to localhost:8000,
see root README's step 5):
    python log_vllm_bench_to_mlflow.py \
        --mlflow-tracking-uri http://localhost:5000 \
        --base-url http://localhost:8000 --model Qwen/Qwen3-8B \
        --dataset-name hf --dataset-path gorilla-llm/Berkeley-Function-Calling-Leaderboard \
        --bfcl-categories simple,multiple,parallel,parallel_multiple \
        --num-prompts 100 --max-concurrency 32

Any extra arguments (e.g. --seed, --request-rate) are passed straight through to
`vllm bench serve` unmodified. Pass --max-concurrency for a single level (e.g. one point
on the 1/8/16/24/32 production comparison, see root README's "Confirmed constraints") --
logged as its own MLflow run, with plain (unprefixed) metric/param names.

To sweep several levels into ONE MLflow run, use --concurrencies (comma-separated)
instead of --max-concurrency -- runs `vllm bench serve` once per value, all logged into
the same run with each value's metrics/params prefixed "c<N>_" (e.g. c32_mean_ttft_ms) so
they sit side by side instead of colliding -- MLflow params are immutable per key, so
logging the same unprefixed key twice with a different value (e.g. "date") would raise
INVALID_PARAMETER_VALUE on the second concurrency, the same class of bug
log_bfcl_to_mlflow.py already had to work around for multi-category runs:
    python log_vllm_bench_to_mlflow.py \
        --mlflow-tracking-uri http://localhost:5000 \
        --base-url http://localhost:8000 --model Qwen/Qwen3-8B \
        --dataset-name hf --dataset-path gorilla-llm/Berkeley-Function-Calling-Leaderboard \
        --bfcl-categories simple,multiple,parallel,parallel_multiple \
        --num-prompts 100 --concurrencies 16,32,64
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

import mlflow
from prometheus_client.parser import text_string_to_metric_families

# vLLM's own KV-cache occupancy gauge, exposed on its OpenAI-compatible server's
# /metrics endpoint (Prometheus text format) by default -- verified against vLLM's
# current source (vllm/v1/metrics/loggers.py) and its own test suite
# (tests/entrypoints/serve/instrumentator/test_metrics.py hits server_url + "/metrics"
# with no extra server flag needed), not assumed. `vllm bench serve` itself never sees
# this -- it only measures client-observed latency/throughput -- so it has to be
# scraped from the server independently, while the benchmark subprocess runs.
KV_CACHE_METRIC_NAME = "vllm:kv_cache_usage_perc"
KV_CACHE_POLL_INTERVAL_S = 1.0


def poll_kv_cache_usage(metrics_url: str, samples: list[float], stop_event: threading.Event) -> None:
    """Background poller: scrapes `metrics_url` every KV_CACHE_POLL_INTERVAL_S seconds
    for KV_CACHE_METRIC_NAME, appending each reading to `samples`. Best-effort --
    an unreachable endpoint (e.g. an older vLLM build, or metrics disabled) just means
    no samples get collected; it must never fail the benchmark subprocess it runs
    alongside."""
    while not stop_event.is_set():
        try:
            with urllib.request.urlopen(metrics_url, timeout=5) as resp:
                text = resp.read().decode()
            for family in text_string_to_metric_families(text):
                if family.name == KV_CACHE_METRIC_NAME:
                    samples.extend(sample.value for sample in family.samples)
        except (urllib.error.URLError, TimeoutError, ValueError):
            pass
        stop_event.wait(KV_CACHE_POLL_INTERVAL_S)

# Per-request raw data in vllm bench serve's --save-result JSON -- large, and not
# scalars, so not loggable (or useful) as MLflow metrics. Matches
# save_to_pytorch_benchmark_format's own ignored_metrics list in vLLM's benchmarks/
# serve.py, plus the other list-valued keys that same file's `result = {...}` /
# `result_json = {...}` construction writes.
LIST_VALUED_KEYS = {
    "input_lens", "output_lens", "ttfts", "itls", "latencies", "start_times",
    "queue_times", "generated_texts", "errors", "rps_change_events",
    "spec_decode_per_position_acceptance_rates",
}

# String/identity fields -- logged as params (run configuration/labels), not metrics
# (MLflow metrics must be numeric).
PARAM_STRING_KEYS = {"date", "endpoint_type", "backend", "label", "model_id", "tokenizer_id"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mlflow-tracking-uri", required=True)
    parser.add_argument("--mlflow-experiment-name", default="fin-agent-vllm-bench")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="openai-chat")
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--dataset-name", default="hf")
    parser.add_argument("--dataset-path", default="gorilla-llm/Berkeley-Function-Calling-Leaderboard")
    parser.add_argument(
        "--bfcl-categories",
        default=None,
        help="Comma-separated, forwarded to vllm bench serve's own --bfcl-categories -- "
        "see root README's step 5 for why simple,multiple,parallel,parallel_multiple "
        "(this project's AST-evaluated 'python' BFCL scope) is the usual choice here.",
    )
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--max-concurrency", type=int, default=None, help="Single concurrency level. Mutually exclusive with --concurrencies.")
    parser.add_argument(
        "--concurrencies",
        default=None,
        help="Comma-separated concurrency levels (e.g. 16,32,64) -- runs the full "
        "benchmark once per value, each its own MLflow run. Mutually exclusive with "
        "--max-concurrency.",
    )
    args, extra_vllm_args = parser.parse_known_args()

    if (args.max_concurrency is None) == (args.concurrencies is None):
        raise SystemExit("pass exactly one of --max-concurrency or --concurrencies")
    concurrencies = (
        [args.max_concurrency] if args.concurrencies is None
        else [int(c) for c in args.concurrencies.split(",") if c.strip()]
    )

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    # One run for the whole sweep -- multi-value only, so a plain single-concurrency
    # call keeps the simple unprefixed metric names it always had.
    prefix_each = len(concurrencies) > 1
    with mlflow.start_run() as run:
        mlflow.log_param("base_url", args.base_url)
        if prefix_each:
            mlflow.set_tag("concurrencies", ",".join(str(c) for c in concurrencies))
        for concurrency in concurrencies:
            prefix = f"c{concurrency}_" if prefix_each else ""
            run_once(args, concurrency, extra_vllm_args, prefix)
        print(f"[log_vllm_bench_to_mlflow] logged to MLflow run {run.info.run_id}")


def run_once(args: argparse.Namespace, concurrency: int, extra_vllm_args: list[str], prefix: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "result.json"
        cmd = [
            "vllm", "bench", "serve",
            "--backend", args.backend,
            "--base-url", args.base_url,
            "--endpoint", args.endpoint,
            "--model", args.model,
            "--dataset-name", args.dataset_name,
            "--dataset-path", args.dataset_path,
            "--num-prompts", str(args.num_prompts),
            "--max-concurrency", str(concurrency),
            "--save-result",
            "--result-dir", str(tmp),
            "--result-filename", result_path.name,
            # vllm bench serve only computes p99 by default (--metric-percentiles
            # defaults to "99") -- p50/p95/p99 here so mean/median/p95/p99 all land in
            # the result JSON (and thus MLflow) without a caller having to remember to
            # ask for it. A caller-supplied --metric-percentiles in {posargs} still
            # wins -- argparse takes the last occurrence of a flag, and extra_vllm_args
            # is appended after this.
            "--metric-percentiles", "50,95,99",
        ]
        if args.bfcl_categories:
            cmd += ["--bfcl-categories", args.bfcl_categories]
        # vLLM's own --metadata mechanism (KEY=VALUE, saved into the result JSON
        # verbatim) -- used here instead of a separate bookkeeping path so
        # dataset_name/dataset_path/bfcl_categories ride along in the one file this
        # script already parses below, the same way num_prompts/max_concurrency do.
        cmd += [
            "--metadata",
            f"dataset_name={args.dataset_name}",
            f"dataset_path={args.dataset_path}",
            f"bfcl_categories={args.bfcl_categories or ''}",
        ]
        cmd += extra_vllm_args

        kv_cache_samples: list[float] = []
        stop_event = threading.Event()
        poller = threading.Thread(
            target=poll_kv_cache_usage,
            args=(f"{args.base_url}/metrics", kv_cache_samples, stop_event),
            daemon=True,
        )
        poller.start()
        try:
            print(f"[log_vllm_bench_to_mlflow] $ {' '.join(cmd)}", file=sys.stderr)
            subprocess.run(cmd, check=True)
        finally:
            stop_event.set()
            poller.join(timeout=KV_CACHE_POLL_INTERVAL_S + 5)

        result = json.loads(result_path.read_text())

    if kv_cache_samples:
        mlflow.log_metric(f"{prefix}kv_cache_usage_perc_mean", sum(kv_cache_samples) / len(kv_cache_samples))
        mlflow.log_metric(f"{prefix}kv_cache_usage_perc_max", max(kv_cache_samples))
    else:
        print(
            f"[log_vllm_bench_to_mlflow] no {KV_CACHE_METRIC_NAME} samples collected from "
            f"{args.base_url}/metrics -- endpoint unreachable, or this vLLM build doesn't expose it",
            file=sys.stderr,
        )

    for key, value in result.items():
        if key in LIST_VALUED_KEYS:
            continue
        if key in PARAM_STRING_KEYS or isinstance(value, str):
            mlflow.log_param(f"{prefix}{key}", value)
        elif isinstance(value, bool) or value is None:
            mlflow.log_param(f"{prefix}{key}", value)
        elif isinstance(value, (int, float)):
            mlflow.log_metric(f"{prefix}{key}", value)
        # anything else (nested dict/list not covered above) is skipped rather than
        # guessed at -- e.g. a future vllm version adding a new structured field.
    if not prefix:
        mlflow.set_tag("max_concurrency", concurrency)
    print(f"[log_vllm_bench_to_mlflow] concurrency={concurrency} logged (prefix={prefix!r})")


if __name__ == "__main__":
    main()
