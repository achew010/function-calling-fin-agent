"""Summarize where a BFCL run actually lost points, from score files already on disk.

`bfcl evaluate` writes {score_dir}/{model}/BFCL_v3_{category}_score.json as JSON lines:
line 0 is {"accuracy", "correct_count", "total_count"}, and every line after it is one
FAILED case carrying {id, error, error_type, prompt, model_result_raw,
model_result_decoded, possible_answer} -- verified against bfcl_eval's own
eval_checker/eval_runner.py, not assumed. That per-failure detail is the useful part and
nothing in this project read it before: log_bfcl_to_mlflow.py only took line 0.

Reads existing results -- no re-run, no GPU. Run it against the same --bfcl-project-root
a previous eval used (the suite script clears that directory at the START of a run, so
the last completed run's results are still there until the next one begins).

Usage:
    python summarize_bfcl_errors.py --score-dir /data/bfcl-sft/score

Reports, in order: per-category accuracy split Non-Live/Live (the two groups the public
leaderboard reports separately, and the split this project compares against), an
error_type histogram for each group, and sample failures for the most common types.
Comparing the two groups' histograms is the point: a fine-tune that helps Non-Live while
hurting Live shows up here as different dominant error types per group, which a single
overall accuracy number hides.

collect()/render_report() are also imported by log_bfcl_to_mlflow.py, which logs the same
error-type counts as MLflow metrics and the same rendered text as a run artifact -- one
implementation, two entry points.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json

# The leaderboard's own Non-Live/Live split -- every "live_"-prefixed category is Live.
# Matches log_bfcl_to_mlflow.py's split_non_live_live().
GROUPS = ("Non-Live", "Live")


def read_score_file(path: Path) -> tuple[dict, list[dict]]:
    """(summary, failures) -- summary is line 0, failures are every line after it."""
    summary: dict = {}
    failures: list[dict] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if i == 0:
                summary = record
            else:
                failures.append(record)
    return summary, failures


def model_score_dir(score_dir: Path, model: str) -> Path:
    """BFCL flattens model ids when naming its directories (Qwen/Qwen3-8B ->
    Qwen_Qwen3-8B), the same quirk read_category_summaries() in log_bfcl_to_mlflow.py
    already has to account for."""
    return score_dir / model.replace("/", "_")


def collect(score_dir: Path, model: str) -> dict[str, list[tuple[str, dict, list[dict]]]]:
    """{group: [(category, summary, failures), ...]} for every category scored so far."""
    directory = model_score_dir(score_dir, model)
    if not directory.is_dir():
        existing = sorted(p.name for p in score_dir.iterdir()) if score_dir.is_dir() else []
        raise FileNotFoundError(f"no score directory at {directory}; {score_dir} contains: {existing}")

    by_group: dict[str, list[tuple[str, dict, list[dict]]]] = defaultdict(list)
    for path in sorted(directory.glob("BFCL_v3_*_score.json")):
        category = path.stem.removeprefix("BFCL_v3_").removesuffix("_score")
        summary, failures = read_score_file(path)
        by_group["Live" if category.startswith("live_") else "Non-Live"].append((category, summary, failures))
    if not by_group:
        raise FileNotFoundError(f"no BFCL_v3_*_score.json files in {directory}")
    return by_group


def error_type_counts(by_group: dict[str, list[tuple[str, dict, list[dict]]]]) -> dict[str, Counter[str]]:
    """{group: Counter({error_type: count})} -- the histogram behind each group's score."""
    counts: dict[str, Counter[str]] = {}
    for group, entries in by_group.items():
        counter: Counter[str] = Counter()
        for _, _, failures in entries:
            for failure in failures:
                counter[failure.get("error_type", "<none>")] += 1
        counts[group] = counter
    return counts


def render_report(by_group: dict[str, list[tuple[str, dict, list[dict]]]], samples: int = 3) -> str:
    lines: list[str] = []
    counts = error_type_counts(by_group)
    for group in GROUPS:
        entries = by_group.get(group)
        if not entries:
            continue
        correct = sum(s.get("correct_count", 0) for _, s, _ in entries)
        total = sum(s.get("total_count", 0) for _, s, _ in entries)
        lines.append("=" * 72)
        lines.append(f" {group} (AST) -- {correct}/{total} = {correct / total:.4f}" if total else f" {group} -- no cases")
        lines.append("=" * 72)
        lines.append(f"{'category':<28}{'accuracy':>10}{'correct':>10}{'total':>8}")
        for category, summary, _ in sorted(entries, key=lambda e: e[1].get("accuracy", 0)):
            lines.append(
                f"{category:<28}{summary.get('accuracy', 0):>10.4f}"
                f"{summary.get('correct_count', 0):>10}{summary.get('total_count', 0):>8}"
            )

        by_error: dict[str, list[dict]] = defaultdict(list)
        for _, _, failures in entries:
            for failure in failures:
                by_error[failure.get("error_type", "<none>")].append(failure)

        group_failures = sum(counts[group].values())
        lines.append("")
        lines.append(f"  failures by error_type ({group_failures} total):")
        for error_type, count in counts[group].most_common():
            lines.append(f"    {count:>5}  {count / group_failures:>6.1%}  {error_type}")

        for error_type, _ in counts[group].most_common(3):
            lines.append("")
            lines.append(f"  --- samples: {error_type} ---")
            for failure in by_error[error_type][:samples]:
                lines.append(f"    id:       {failure.get('id')}")
                lines.append(f"    got:      {str(failure.get('model_result_decoded'))[:300]}")
                lines.append(f"    expected: {str(failure.get('possible_answer'))[:300]}")
                lines.append(f"    error:    {str(failure.get('error'))[:300]}")
                lines.append("")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score-dir", type=Path, required=True, help="e.g. /data/bfcl-sft/score")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="Flattened the same way BFCL names its own directories (Qwen/Qwen3-8B -> Qwen_Qwen3-8B).",
    )
    parser.add_argument("--samples", type=int, default=3, help="Sample failures to print per dominant error type.")
    args = parser.parse_args()

    print(render_report(collect(args.score_dir, args.model), samples=args.samples))


if __name__ == "__main__":
    main()
