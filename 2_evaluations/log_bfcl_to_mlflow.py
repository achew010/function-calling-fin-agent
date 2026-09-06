"""Run BFCLEvaluator and log its per-category summary metrics to MLflow.

bfcl-eval itself has no MLflow integration — this bridges the gap. It writes one JSONL
score file per test category at
{BFCL_PROJECT_ROOT}/score/{model}/BFCL_v3_{category}_score.json, whose first line is
always {"accuracy": ..., "correct_count": ..., "total_count": ...} (everything after
that line is per-failure detail) — verified against the installed bfcl-eval package's
eval_runner.py/utils.py source, not assumed.

Usage (against an already-running, OpenAI-compatible vLLM server):
    python log_bfcl_to_mlflow.py \
        --model Qwen/Qwen3-8B --test-category python \
        --skip-server-setup --vllm-endpoint <host> --vllm-port 8000 \
        --mlflow-tracking-uri http://mlflow.apps.svc.cluster.local:5000

This only logs final, whole-suite metrics once generate+evaluate+scores all finish,
which can take hours. To also see progress and per-category scores while it's still
running, start poll_bfcl_progress.py alongside this process, pointed at the same
--bfcl-project-root and this run's id (pass --run-id-file here so the poller can pick
it up as soon as the run starts).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mlflow

from run_bfcl_eval import BFCLEvaluator


def read_category_summaries(score_dir: Path, model: str) -> dict[str, dict]:
    model_dir = score_dir / model
    summaries: dict[str, dict] = {}
    if not model_dir.exists():
        return summaries
    for f in sorted(model_dir.glob("BFCL_v3_*_score.json")):
        category = f.stem.removeprefix("BFCL_v3_").removesuffix("_score")
        with f.open() as fh:
            first_line = fh.readline()
        if first_line:
            summaries[category] = json.loads(first_line)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--test-category", default="python")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Concurrent request threads BFCL's own harness uses against the server — "
        "the default of 1 sends requests serially regardless of the server's batching "
        "capacity. See run_bfcl_eval.py's --num-threads help for more.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--backend", default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--local-model-path", default=None)
    parser.add_argument("--skip-server-setup", action="store_true")
    parser.add_argument("--vllm-endpoint", default="localhost")
    parser.add_argument("--vllm-port", default="8000")
    parser.add_argument(
        "--bfcl-project-root",
        default="/tmp/bfcl",
        help="Sets BFCL_PROJECT_ROOT so result/score files land somewhere known and writable "
        "instead of bfcl-eval's own package install directory (its default).",
    )
    parser.add_argument("--mlflow-tracking-uri", required=True)
    parser.add_argument("--mlflow-experiment-name", default="fin-agent-bfcl")
    parser.add_argument(
        "--run-id-file",
        default=None,
        help="Write the MLflow run id here as soon as the run starts (before generate/"
        "evaluate run), so a concurrently-started poll_bfcl_progress.py process can "
        "attach to the same run via --mlflow-run-id.",
    )
    args = parser.parse_args()

    os.environ["BFCL_PROJECT_ROOT"] = args.bfcl_project_root
    score_dir = Path(args.bfcl_project_root) / "score"

    evaluator = BFCLEvaluator(
        model=args.model,
        test_category=args.test_category,
        num_gpus=args.num_gpus,
        num_threads=args.num_threads,
        gpu_memory_utilization=args.gpu_memory_utilization,
        backend=args.backend,
        local_model_path=args.local_model_path,
        skip_server_setup=args.skip_server_setup,
        vllm_endpoint=args.vllm_endpoint,
        vllm_port=args.vllm_port,
    )
    eval_dataset = evaluator.load_eval_dataset()
    print(f"Evaluating {evaluator.model} on {len(eval_dataset)} test file(s): {list(eval_dataset)}")

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    # If --run-id-file already holds an id (e.g. this is a Job retry after a killed pod,
    # and the file lives on the same persistent volume as the bfcl results it names),
    # resume logging into that same run instead of starting a new one each attempt.
    existing_run_id = None
    if args.run_id_file and Path(args.run_id_file).exists():
        existing_run_id = Path(args.run_id_file).read_text().strip() or None

    with mlflow.start_run(run_id=existing_run_id) as run:
        # Written (on first attempt only) before generate/evaluate runs, which can take
        # hours, so poll_bfcl_progress.py started alongside this process -- and any
        # retry of this process after a pod restart -- can attach to the same run.
        if args.run_id_file:
            Path(args.run_id_file).write_text(run.info.run_id)
        mlflow.log_param("model", args.model)
        mlflow.log_param("test_category", args.test_category)
        mlflow.log_param("backend", args.backend)
        mlflow.log_param("num_threads", args.num_threads)

        predictions = evaluator.run_predictions(eval_dataset)
        evaluator.compute_metrics(predictions)

        summaries = read_category_summaries(score_dir, args.model)
        print(json.dumps(summaries, indent=2))

        for category, summary in summaries.items():
            mlflow.log_metric(f"bfcl_{category}_accuracy", summary.get("accuracy", 0.0))
            mlflow.log_metric(f"bfcl_{category}_correct_count", summary.get("correct_count", 0))
            mlflow.log_metric(f"bfcl_{category}_total_count", summary.get("total_count", 0))
        total_correct = sum(s.get("correct_count", 0) for s in summaries.values())
        total_count = sum(s.get("total_count", 0) for s in summaries.values())
        if total_count:
            mlflow.log_metric("bfcl_overall_accuracy", total_correct / total_count)
        mlflow.log_metric("bfcl_categories_evaluated", len(summaries))


if __name__ == "__main__":
    main()
