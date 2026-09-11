"""Background poller: logs BFCL generation progress + per-category scores to an
existing MLflow run at intervals, while `bfcl generate` (driven by run_bfcl_eval.py /
log_bfcl_to_mlflow.py) is still running in a separate process against the same
--bfcl-project-root.

Why this has to work off raw line counts rather than `bfcl evaluate`: bfcl-eval (pinned
2025.8.6.2) writes each completed test case as its own JSONL line to
result/<model>/BFCL_v3_<category>_result.json as soon as it finishes
(base_handler.py's write(), append mode, one open/write/close per case) — but
`bfcl evaluate` hard-asserts the result file's line count matches the category's full
test count and crashes (AssertionError) on anything still in progress. So this script
polls completed-line counts for live progress on every category, and only invokes
`bfcl evaluate` for a category once its line count reaches that category's known total
(read directly from bfcl-eval's own bundled test file under PROMPT_PATH) — never on a
still-in-progress one. Deliberately never calls `bfcl scores` afterward — see the
scoring loop below and run_bfcl_eval.py's compute_metrics() for why.

Usage: run alongside (not instead of) run_bfcl_eval.py/log_bfcl_to_mlflow.py, pointed
at the same --bfcl-project-root and an MLflow run id that process already created:
    python poll_bfcl_progress.py --model Qwen/Qwen3-8B --test-category python \
        --bfcl-project-root /data/bfcl --mlflow-tracking-uri http://... \
        --mlflow-run-id <run-id> --interval 60 --stop-file /tmp/bfcl-poll-stop
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import mlflow


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--test-category", default="python")
    parser.add_argument("--bfcl-project-root", default="/tmp/bfcl")
    parser.add_argument("--mlflow-tracking-uri", required=True)
    parser.add_argument("--mlflow-experiment-name", default="fin-agent-bfcl")
    parser.add_argument(
        "--mlflow-run-id",
        required=True,
        help="Attach to this existing run (started by log_bfcl_to_mlflow.py via "
        "--run-id-file) instead of creating a new one.",
    )
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between polls.")
    parser.add_argument(
        "--stop-file",
        default=None,
        help="Exit (after one final poll) once this file appears. Without it the "
        "poller runs until killed.",
    )
    args = parser.parse_args()

    os.environ["BFCL_PROJECT_ROOT"] = args.bfcl_project_root

    # Imported only after BFCL_PROJECT_ROOT is set: eval_config.py resolves
    # RESULT_PATH/SCORE_PATH from that env var at *import* time (and caches the result
    # in sys.modules for the rest of the process), same as run_bfcl_eval.py /
    # log_bfcl_to_mlflow.py. read_category_summaries has to be deferred here too, not
    # imported at module level like it originally was -- log_bfcl_to_mlflow imports
    # run_bfcl_eval, which imports bfcl_eval.constants.*, so a top-level import of it
    # would pull in (and permanently cache) eval_config's RESULT_PATH/SCORE_PATH before
    # BFCL_PROJECT_ROOT above was ever set, silently pinning this whole process to
    # bfcl-eval's default paths instead of --bfcl-project-root. Hit this for real: every
    # poll reported 0 completed cases despite bfcl generate actively writing results,
    # because this process was looking in the wrong directory the entire time.
    from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING, TEST_FILE_MAPPING
    from bfcl_eval.constants.eval_config import PROMPT_PATH, RESULT_PATH, SCORE_PATH
    from bfcl_eval.utils import load_file
    from log_bfcl_to_mlflow import read_category_summaries

    categories = (
        TEST_COLLECTION_MAPPING[args.test_category]
        if args.test_category in TEST_COLLECTION_MAPPING
        else [args.test_category]
    )
    expected_counts = {cat: len(load_file(PROMPT_PATH / TEST_FILE_MAPPING[cat])) for cat in categories}
    model_dir = RESULT_PATH / args.model.replace("/", "_")

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    scored: set[str] = set()
    step = 0
    stop_file = Path(args.stop_file) if args.stop_file else None

    while True:
        completed_counts = {
            cat: count_lines(model_dir / f"BFCL_v3_{cat}_result.json") for cat in categories
        }
        newly_complete = [
            cat
            for cat, n in completed_counts.items()
            if cat not in scored and expected_counts[cat] and n >= expected_counts[cat]
        ]
        for cat in newly_complete:
            print(f"[poll] {cat} fully generated ({completed_counts[cat]}/{expected_counts[cat]}) -- scoring")
            try:
                # `bfcl evaluate` alone writes BFCL_v3_<cat>_score.json, which is all
                # read_category_summaries() below needs. Deliberately not calling
                # `bfcl scores` afterward -- see run_bfcl_eval.py's compute_metrics()
                # docstring: it crashes (ValueError: 'Non-Live Exec Acc' is not in
                # list) whenever exec_* categories weren't run, which is always true
                # for this project's "python" test-category scope.
                subprocess.run(["bfcl", "evaluate", "--model", args.model, "--test-category", cat], check=True)
            except subprocess.CalledProcessError as e:
                print(f"[poll] scoring {cat} failed, will retry next poll: {e}")
                continue
            scored.add(cat)

        total_completed = sum(completed_counts.values())
        total_expected = sum(expected_counts.values())
        metrics = {f"gen_completed_{cat}": n for cat, n in completed_counts.items()}
        metrics.update(
            {f"gen_progress_{cat}": n / expected_counts[cat] for cat, n in completed_counts.items() if expected_counts[cat]}
        )
        metrics["gen_completed_overall"] = total_completed
        if total_expected:
            metrics["gen_progress_overall"] = total_completed / total_expected

        summaries = read_category_summaries(SCORE_PATH, args.model)
        for cat, summary in summaries.items():
            metrics[f"bfcl_{cat}_accuracy"] = summary.get("accuracy", 0.0)
            metrics[f"bfcl_{cat}_correct_count"] = summary.get("correct_count", 0)
            metrics[f"bfcl_{cat}_total_count"] = summary.get("total_count", 0)

        with mlflow.start_run(run_id=args.mlflow_run_id):
            mlflow.log_metrics(metrics, step=step)
        print(f"[poll] step={step} completed={total_completed}/{total_expected} scored={sorted(scored)}")
        step += 1

        if stop_file and stop_file.exists():
            print("[poll] stop file detected, exiting after final poll")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
