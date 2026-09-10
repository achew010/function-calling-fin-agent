"""Run `vllm bench serve` and log its scalar metrics (TTFT, TPOT, ITL, throughput, ...)
to MLflow -- vllm bench serve itself has no MLflow integration, only its own local
--save-result JSON file. This bridges that gap the same way log_bfcl_to_mlflow.py bridges
it for bfcl-eval: run the real tool as a subprocess, point --save-result at a temp file,
and log every scalar key from the result JSON as an MLflow metric or param.

Key names logged as metrics -- verified against vLLM's own benchmarks/serve.py source,
not assumed -- include mean/median/std/p99 for ttft_ms, tpot_ms, itl_ms, e2el_ms, plus
request_throughput, output_throughput, total_token_throughput, completed, failed. Per-
request raw lists (ttfts, itls, latencies, generated_texts, errors, ...) are skipped --
MLflow metrics must be scalars, and these are exactly the detail save_to_pytorch_
benchmark_format's own ignored_metrics list already excludes for the same reason.

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
`vllm bench serve` unmodified. Run once per concurrency level you want to compare (e.g.
1/8/16/24/32, see root README's "Confirmed constraints") -- each invocation is logged as
its own MLflow run so they show up as separate, directly comparable rows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow

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
    parser.add_argument("--max-concurrency", type=int, required=True)
    args, extra_vllm_args = parser.parse_known_args()

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
            "--max-concurrency", str(args.max_concurrency),
            "--save-result",
            "--result-dir", str(tmp),
            "--result-filename", result_path.name,
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

        print(f"[log_vllm_bench_to_mlflow] $ {' '.join(cmd)}", file=sys.stderr)
        subprocess.run(cmd, check=True)

        result = json.loads(result_path.read_text())

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_param("base_url", args.base_url)
        for key, value in result.items():
            if key in LIST_VALUED_KEYS:
                continue
            if key in PARAM_STRING_KEYS or isinstance(value, str):
                mlflow.log_param(key, value)
            elif isinstance(value, bool) or value is None:
                mlflow.log_param(key, value)
            elif isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
            # anything else (nested dict/list not covered above) is skipped rather than
            # guessed at -- e.g. a future vllm version adding a new structured field.
        mlflow.set_tag("max_concurrency", args.max_concurrency)
        print(f"[log_vllm_bench_to_mlflow] logged to MLflow run {run.info.run_id}")


if __name__ == "__main__":
    main()
