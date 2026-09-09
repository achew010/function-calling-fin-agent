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
from importlib.metadata import version as pkg_version
from pathlib import Path

import mlflow

from run_bfcl_eval import BFCLEvaluator

# The public leaderboard's own headline "Overall Acc" is a composite across five domains
# (Agentic, Multi-Turn, Single-Turn/AST, Hallucination, Format Sensitivity), with Agentic
# alone weighted 40% -- verified directly against the live leaderboard
# (gorilla.cs.berkeley.edu) and its BFCL_v4 changelog. This project's "python" test
# category only ever covers the Single-Turn/AST domain (bfcl-eval==2025.8.6.2 predates
# BFCL_v4's Agentic categories entirely -- Web Search and Memory Management didn't exist
# yet), split the same way the leaderboard itself splits it: every "live_*" category is
# Live (AST), everything else in our set is Non-Live (AST). Comparing our flat average
# across all 11 categories against the leaderboard's "Overall Acc" column is comparing
# two different statistics -- e.g. Qwen3-8B (Prompt) shows "Overall Acc" 40.43 on the
# live leaderboard (dragged down by ~13% Agentic accuracy) but "Non-Live (AST)" 88.56 /
# "Live (AST)" 80.09 -- it's those two columns this project's numbers are actually
# comparable to.
BFCL_COMPARABILITY_NOTE = (
    "Only Non-Live/Live AST (Single-Turn) categories are evaluated here -- no Agentic "
    "(Web Search/Memory), Multi-Turn, or Hallucination coverage, since those either "
    "didn't exist yet in the pinned bfcl-eval version or aren't in this project's "
    "'python' test-category scope. Compare bfcl_non_live_ast_accuracy / "
    "bfcl_live_ast_accuracy against the public leaderboard's 'Non-Live (AST)' / "
    "'Live (AST)' columns for the matching model+variant row -- NOT against the "
    "leaderboard's composite 'Overall Acc' column, which weights Agentic accuracy at "
    "40% and will read much lower for any model (including this one) that domain was "
    "never evaluated on."
)

# Qwen/Qwen3-8B's own leaderboard rows, read directly off gorilla.cs.berkeley.edu (page's
# own "Last Updated: 2026-04-12" stamp) -- logged as metrics (not just params) below,
# under the SAME names as our own bfcl_non_live_ast_accuracy/bfcl_live_ast_accuracy but
# prefixed "leaderboard_", specifically so both sit side by side as directly-comparable
# numeric columns in MLflow's own metrics table/run-comparison view, not just documented
# in a note someone has to go read. Re-verify against the live page if it's been a while
# -- "will be updated periodically" per the leaderboard's own description -- and update
# these two entries; nothing else in this file needs to change if the leaderboard adds or
# reorders other models.
LEADERBOARD_QWEN3_8B_REFERENCE = {
    "Prompt": {"non_live_ast": 0.8856, "live_ast": 0.8009},
    "FC": {"non_live_ast": 0.8758, "live_ast": 0.8053},
}


def read_category_summaries(score_dir: Path, model: str) -> dict[str, dict]:
    # BFCL flattens model ids when naming result/score directories. Using the raw
    # Hugging Face id here creates score/Qwen/Qwen3-8B, which never exists; the actual
    # directory is score/Qwen_Qwen3-8B.
    model_dir = score_dir / model.replace("/", "_")
    summaries: dict[str, dict] = {}
    if not model_dir.exists():
        existing = sorted(p.name for p in score_dir.iterdir()) if score_dir.exists() else []
        raise FileNotFoundError(
            f"BFCL produced no score directory at {model_dir}; "
            f"{score_dir} contains: {existing}"
        )
    for f in sorted(model_dir.glob("BFCL_v3_*_score.json")):
        category = f.stem.removeprefix("BFCL_v3_").removesuffix("_score")
        with f.open() as fh:
            first_line = fh.readline()
        if first_line:
            summaries[category] = json.loads(first_line)
    if not summaries:
        raise RuntimeError(f"BFCL produced no category summaries in {model_dir}")
    return summaries


def group_accuracy(group: dict[str, dict]) -> tuple[float | None, int, int]:
    """(accuracy, correct_count, total_count) for a group of category summaries --
    None accuracy (rather than 0.0) when the group is empty, so callers can distinguish
    "not yet run" from "ran and got every case wrong"."""
    correct = sum(s.get("correct_count", 0) for s in group.values())
    total = sum(s.get("total_count", 0) for s in group.values())
    return (correct / total if total else None), correct, total


def split_non_live_live(summaries: dict[str, dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    """Matches the public leaderboard's own Non-Live (AST) / Live (AST) grouping: every
    BFCL category prefixed "live_" is Live, everything else (simple/irrelevance/
    parallel/multiple/parallel_multiple) is Non-Live -- verified against the live
    leaderboard's category naming, not assumed."""
    live = {c: s for c, s in summaries.items() if c.startswith("live_")}
    non_live = {c: s for c, s in summaries.items() if not c.startswith("live_")}
    return non_live, live


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
        model_variant = "FC" if evaluator.is_fc_model else "Prompt"
        # Params are IMMUTABLE in MLflow, and this script is invoked once per test
        # category against the same run whenever a caller shares one --run-id-file
        # across categories (see bfcl-eval-*-job.yaml's category loop, which does that
        # so one run ends up holding both Non-Live and Live AST accuracy). Re-logging
        # test_category on the second category raised INVALID_PARAMETER_VALUE and killed
        # the job after the first category had already finished. Every param here is
        # identical across categories except test_category, so log them only when this
        # process actually started the run; a plain retry of the same category was always
        # fine (MLflow allows re-logging an identical value) but a second category is not.
        if existing_run_id is None:
            mlflow.log_param("model", args.model)
            # Kept so a single-category run records exactly what it always did. When a
            # run spans several categories the plural `test_categories` tag below is the
            # authoritative one -- this param holds whichever category ran first.
            mlflow.log_param("test_category", args.test_category)
            mlflow.log_param("backend", args.backend)
            mlflow.log_param("num_threads", args.num_threads)
            # Which leaderboard row this run is actually the analogue of (FC and Prompt
            # are tracked as separate rows with different scores, e.g. Qwen3-8B (FC) vs.
            # Qwen3-8B (Prompt)) and which bfcl-eval build produced these numbers -- both
            # necessary to interpret this run at all, months later or by someone else.
            mlflow.log_param("model_variant", model_variant)
            try:
                mlflow.log_param("bfcl_eval_version", pkg_version("bfcl-eval"))
            except Exception:
                pass  # best-effort -- never fail the run over a version-string lookup
            mlflow.log_param("bfcl_comparability_note", BFCL_COMPARABILITY_NOTE)
        # Tags, unlike params, can be updated -- so the categories this run actually
        # covers accumulate here rather than being pinned to whichever one ran first.
        seen = mlflow.get_run(run.info.run_id).data.tags.get("test_categories", "")
        covered = [c for c in seen.split(",") if c]
        if args.test_category not in covered:
            covered.append(args.test_category)
        mlflow.set_tag("test_categories", ",".join(covered))

        predictions = evaluator.run_predictions(eval_dataset)
        evaluator.compute_metrics(predictions)

        summaries = read_category_summaries(score_dir, args.model)
        missing_categories = set(eval_dataset) - set(summaries)
        if missing_categories:
            raise RuntimeError(
                "BFCL did not produce scores for: " + ", ".join(sorted(missing_categories))
            )
        print(json.dumps(summaries, indent=2))

        for category, summary in summaries.items():
            mlflow.log_metric(f"bfcl_{category}_accuracy", summary.get("accuracy", 0.0))
            mlflow.log_metric(f"bfcl_{category}_correct_count", summary.get("correct_count", 0))
            mlflow.log_metric(f"bfcl_{category}_total_count", summary.get("total_count", 0))

        # The two numbers actually comparable to the public leaderboard -- see
        # BFCL_COMPARABILITY_NOTE above for why bfcl_overall_accuracy below is not.
        non_live, live = split_non_live_live(summaries)
        non_live_acc, non_live_correct, non_live_total = group_accuracy(non_live)
        live_acc, live_correct, live_total = group_accuracy(live)
        if non_live_acc is not None:
            mlflow.log_metric("bfcl_non_live_ast_accuracy", non_live_acc)
            mlflow.log_metric("bfcl_non_live_ast_correct_count", non_live_correct)
            mlflow.log_metric("bfcl_non_live_ast_total_count", non_live_total)
        if live_acc is not None:
            mlflow.log_metric("bfcl_live_ast_accuracy", live_acc)
            mlflow.log_metric("bfcl_live_ast_correct_count", live_correct)
            mlflow.log_metric("bfcl_live_ast_total_count", live_total)

        # The public leaderboard's own reference numbers for this exact model+variant,
        # logged as metrics (not just documented in BFCL_COMPARABILITY_NOTE) so they show
        # up as directly-comparable numeric columns right alongside
        # bfcl_non_live_ast_accuracy/bfcl_live_ast_accuracy in MLflow's own metrics
        # table/run-comparison view -- no need to go find and re-read this file or the
        # live leaderboard page just to eyeball how close a run came.
        leaderboard_ref = LEADERBOARD_QWEN3_8B_REFERENCE.get(model_variant)
        if leaderboard_ref:
            mlflow.log_metric("leaderboard_non_live_ast_accuracy", leaderboard_ref["non_live_ast"])
            mlflow.log_metric("leaderboard_live_ast_accuracy", leaderboard_ref["live_ast"])

        # Kept for detail, but NOT the number to compare against the leaderboard with --
        # it's a flat average across only this project's 11-category subset, not the
        # leaderboard's own weighted, five-domain "Overall Acc" composite.
        total_correct = sum(s.get("correct_count", 0) for s in summaries.values())
        total_count = sum(s.get("total_count", 0) for s in summaries.values())
        if total_count:
            mlflow.log_metric("bfcl_overall_accuracy", total_correct / total_count)
        mlflow.log_metric("bfcl_categories_evaluated", len(summaries))


if __name__ == "__main__":
    main()
