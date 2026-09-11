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
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

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


def sub_error_types(failure: dict) -> list[str]:
    """The real causes hiding inside a composite failure.

    The parallel checkers report a top-level error_type of
    "parallel_function_checker_no_order:cannot_find_match" and then, for every model call
    they tried to match, append {"Model Result Index N": {"sub_error_type": ..., ...}} to
    the `error` list (bfcl_eval's ast_checker.py builds exactly this). The top-level type
    only says "nothing matched"; the sub_error_type says WHY -- missing_optional vs
    value_error:string vs type_error:simple are three unrelated fixes that all surface as
    cannot_find_match. Counting only the wrapper makes every parallel failure look
    identical.
    """
    found: list[str] = []
    for entry in failure.get("error", []) or []:
        if not isinstance(entry, dict):
            continue  # the leading "Could not find a matching function..." string
        for detail in entry.values():
            if isinstance(detail, dict) and detail.get("sub_error_type"):
                found.append(detail["sub_error_type"])
    return found


def root_cause_counts(by_group: dict[str, list[tuple[str, dict, list[dict]]]]) -> dict[str, Counter[str]]:
    """{group: Counter({cause: count})}, counting each failure ONCE by its real cause.

    A failure with sub-errors is attributed to its distinct sub_error_types (a composite
    failure can legitimately have tried several model calls and failed each differently);
    one without is attributed to its own error_type. This is the histogram to act on --
    error_type_counts() above is the raw view.
    """
    counts: dict[str, Counter[str]] = {}
    for group, entries in by_group.items():
        counter: Counter[str] = Counter()
        for _, _, failures in entries:
            for failure in failures:
                causes = sorted(set(sub_error_types(failure)))
                for cause in causes or [failure.get("error_type", "<none>")]:
                    counter[cause] += 1
        counts[group] = counter
    return counts


# What each cause actually means and what to do about it. Keys are BFCL's own
# error_type/sub_error_type strings, taken from bfcl_eval's eval_checker source
# (ast_checker.py / eval_runner.py), not invented. render_diagnosis() emits entries only
# for causes a run actually hit, so the document stays a worklist rather than a catalog.
REMEDIES: dict[str, tuple[str, str]] = {
    "ast_decoder:decoder_failed": (
        "The response could not be parsed as `[func(arg=val)]` at all.",
        "Look at `got:` in the samples. Prose or a `<think>` tag means the prompt-side fix "
        "regressed -- check patch_bfcl_qwen_handler.py actually ran after `pip install` in "
        "the Job (it patches the installed package on disk; an in-process monkeypatch does "
        "NOT reach the `bfcl generate` subprocess). JSON instead of call syntax means the "
        "SFT targets drifted -- check render_calls() in 0_data/prepare_dataset.py.",
    ),
    "ast_decoder:decoder_wrong_output_format": (
        "Parsed, but not as a list of function calls.",
        "Same prompt/target-format checks as decoder_failed; compare a sample against the "
        "`[func(a=1), func(b=2)]` shape BFCL expects.",
    ),
    "irrelevance_error:decoder_success": (
        "The model emitted a call when the correct behaviour was to abstain.",
        "Over-eagerness to call. Cross-check eval_refusal_accuracy and "
        "eval_error_rate_unwarranted_call on the source training run: if those look fine "
        "on ToolACE-style refusals but this is high, the refusal behaviour isn't "
        "transferring, and no-call coverage in the training mix is the lever.",
    ),
    "relevance_error:decoder_failed": (
        "A call was required and the model produced none (or an unparseable one).",
        "Opposite failure to irrelevance -- check whether the model is over-refusing, and "
        "whether the response was simply unparseable (see decoder_failed).",
    ),
    "simple_function_checker:wrong_func_name": (
        "Wrong tool chosen from the candidate list.",
        "Tool-selection quality. Cross-check eval_tool_selection_exact_match on the source "
        "training run; if that is high while this is too, the candidate lists at eval time "
        "are larger/messier than training saw.",
    ),
    "simple_function_checker:wrong_count": (
        "Wrong NUMBER of calls emitted.",
        "In this repo the recurring shape is collapsing N parallel calls into ONE call with "
        "list-valued arguments (e.g. `f(movie=['A','B'])` where two `f(...)` calls were "
        "expected). Check how parallel calls are represented in the prepared training data "
        "before touching hyperparameters -- this is a data-shape problem, not an LR problem.",
    ),
    "multiple_function_checker:wrong_count": (
        "Wrong NUMBER of calls emitted (multiple-function category).",
        "Same as simple_function_checker:wrong_count.",
    ),
    "parallel_function_checker_no_order:wrong_count": (
        "Wrong NUMBER of parallel calls emitted.",
        "Same as simple_function_checker:wrong_count -- most often N calls collapsed into "
        "one call with array arguments.",
    ),
    "parallel_function_checker_enforce_order:wrong_count": (
        "Wrong NUMBER of parallel calls emitted (ordered variant).",
        "Same as simple_function_checker:wrong_count.",
    ),
    "parallel_function_checker_no_order:cannot_find_match": (
        "Composite wrapper: no model call matched an expected call.",
        "NOT a root cause -- read the root-cause table above, which unpacks the nested "
        "sub_error_type. If this appears as a root cause, the failures carried no "
        "sub-errors, which is unusual and worth inspecting in bfcl_scores/ directly.",
    ),
    "simple_function_checker:missing_required": (
        "A required parameter was omitted.",
        "Under-specification. Check whether the omitted params are ones ToolACE typically "
        "leaves implicit.",
    ),
    "simple_function_checker:missing_optional": (
        "A parameter BFCL's answer key requires to be PRESENT was omitted.",
        "BFCL only tolerates omission when the answer key lists '' as an accepted value. "
        "The model is being conservative about optional arguments; the expected answer in "
        "the sample shows exactly which argument was wanted.",
    ),
    "simple_function_checker:unexpected_param": (
        "A parameter not in the function schema was emitted.",
        "Argument hallucination -- the model invented a parameter name. Cross-check "
        "eval_error_rate_extra_param on the source training run.",
    ),
    "type_error:simple": (
        "Right parameter, wrong type (e.g. '5' where 5 was expected).",
        "Cross-check eval_param_type_accuracy on the source training run. If that metric "
        "DEGRADED over training, SFT is teaching the wrong argument conventions.",
    ),
    "type_error:nested": (
        "Wrong type inside a nested structure.",
        "Same as type_error:simple, but inside a list/dict argument.",
    ),
    "value_error:string": (
        "Right parameter, wrong string value.",
        "Frequently canonicalisation rather than comprehension: 'San Francisco' where "
        "'San Francisco, CA' was expected, or answering in the query's own language. Check "
        "the samples before concluding the model misunderstood the request.",
    ),
    "value_error:dict_value": ("Wrong value inside a dict argument.", "Inspect the sample's expected vs got dicts."),
    "value_error:dict_key": ("Wrong/missing key in a dict argument.", "Inspect the sample's expected vs got dicts."),
    "value_error:list/tuple": ("Wrong list/tuple contents.", "Check element order and count against the sample."),
    "value_error:list_dict_count": ("Wrong number of dicts in a list argument.", "Check the sample's expected list length."),
    "value_error:others": ("Value mismatch not covered by a more specific checker.", "Inspect the sample directly."),
}

# Below this many scored cases a group's accuracy cannot be meaningfully compared against
# the public leaderboard's own figures (those are computed over thousands of cases across
# every category in the group). The smoke pair in run-bfcl-eval-run-ids-suite.sh scores
# just 16 live_parallel cases, where a single case moves the score by >6 points -- a real
# conclusion was drawn from exactly that once, hence the automatic warning.
SMALL_SAMPLE_THRESHOLD = 200


def render_diagnosis(
    by_group: dict[str, list[tuple[str, dict, list[dict]]]],
    model: str,
    source_run_id: str | None = None,
) -> str:
    """A worklist for whoever (or whatever) picks this run up next.

    Ordered so the first thing read is whether the numbers can be trusted at all, then
    where the losses are by real cause, then what to do about each cause present.
    """
    causes = root_cause_counts(by_group)
    lines = [f"# BFCL diagnosis -- {model}", ""]
    lines.append(f"- source training run: `{source_run_id}`" if source_run_id else "- source training run: none (untuned baseline)")

    categories = [c for entries in by_group.values() for c, _, _ in entries]
    lines.append(f"- categories scored: {', '.join(sorted(categories))}")
    lines.append("")

    lines.append("## 1. Are these numbers comparable?")
    lines.append("")
    warned = False
    for group in GROUPS:
        entries = by_group.get(group)
        if not entries:
            continue
        correct = sum(s.get("correct_count", 0) for _, s, _ in entries)
        total = sum(s.get("total_count", 0) for _, s, _ in entries)
        lines.append(f"- **{group} (AST): {correct}/{total} = {correct / total:.4f}**" if total else f"- {group}: no cases")
        if total and total < SMALL_SAMPLE_THRESHOLD:
            warned = True
            lines.append(
                f"  - **WARNING: {total} cases is too few to compare against the leaderboard reference.** "
                f"One case moves this score by {1 / total:.3f}. This looks like a smoke run "
                f"(`run-bfcl-eval-run-ids-suite.sh` defaults to `parallel live_parallel`), and a smoke "
                f"run also omits whole categories -- notably live_irrelevance, where abstention is tested. "
                f"Re-run with `FULL_SCALE=1` before claiming parity or regression either way."
            )
    if not warned:
        lines.append("- Sample sizes are large enough to compare; still prefer a same-harness `baseline` run over leaderboard figures.")
    lines.append("")

    lines.append("## 2. Where the losses actually are (by root cause)")
    lines.append("")
    lines.append("Composite `cannot_find_match` wrappers are unpacked into the nested `sub_error_type` they hide.")
    lines.append("")
    for group in GROUPS:
        counter = causes.get(group)
        if not counter:
            continue
        total_failures = sum(counter.values())
        lines.append(f"### {group} -- {total_failures} failures")
        lines.append("")
        lines.append("| count | share | root cause |")
        lines.append("|---:|---:|---|")
        for cause, count in counter.most_common():
            lines.append(f"| {count} | {count / total_failures:.1%} | `{cause}` |")
        lines.append("")

    lines.append("## 3. What to do about each cause present")
    lines.append("")
    seen: set[str] = set()
    ranked = Counter()
    for counter in causes.values():
        ranked.update(counter)
    for cause, count in ranked.most_common():
        if cause in seen:
            continue
        seen.add(cause)
        meaning, action = REMEDIES.get(cause, ("No remedy entry for this cause yet.", "Inspect samples in bfcl_error_summary.txt and the raw bfcl_scores/ artifact."))
        lines.append(f"### `{cause}` ({count} failures)")
        lines.append("")
        lines.append(f"- **Means:** {meaning}")
        lines.append(f"- **Do:** {action}")
        lines.append("")

    lines.append("## 4. Where to look next")
    lines.append("")
    lines.append("- `bfcl_error_summary.txt` -- per-category accuracy and sample failures (got vs expected) per error type.")
    lines.append("- `bfcl_scores/` -- every failed case in full, including the prompt and the checker's own error list.")
    lines.append("- `bfcl_generations/` -- every raw generation, including the ones that passed.")
    lines.append("- The source training run's own curves (eval_param_type_accuracy, eval_tool_selection_exact_match, eval_refusal_accuracy) -- several remedies above say which one to cross-check.")
    return "\n".join(lines)


def categories_csv(by_group: dict[str, list[tuple[str, dict, list[dict]]]]) -> str:
    """Per-category scores as CSV -- MLflow's artifact viewer renders .csv as a sortable
    table, which the .md/.txt reports can't be (and the raw score files can't either:
    they're JSON Lines, so a viewer that expects one JSON document per file won't parse
    them)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["group", "category", "accuracy", "correct_count", "total_count"])
    for group in GROUPS:
        for category, summary, _ in by_group.get(group, []):
            writer.writerow([
                group, category,
                f"{summary.get('accuracy', 0):.4f}",
                summary.get("correct_count", 0),
                summary.get("total_count", 0),
            ])
    return buffer.getvalue()


def failures_csv(by_group: dict[str, list[tuple[str, dict, list[dict]]]], max_field: int = 500) -> str:
    """One row per failed case, sortable/filterable in MLflow's artifact viewer.

    `root_cause` is the unpacked sub_error_type where there is one, so the table can be
    sorted by the thing actually worth fixing rather than by the cannot_find_match
    wrapper. Long fields are truncated -- the untruncated record is in bfcl_scores/.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["group", "category", "id", "error_type", "root_cause", "model_output", "expected", "error"])
    for group in GROUPS:
        for category, _, failures in by_group.get(group, []):
            for failure in failures:
                causes = sorted(set(sub_error_types(failure)))
                writer.writerow([
                    group, category, failure.get("id"),
                    failure.get("error_type", ""),
                    "; ".join(causes) or failure.get("error_type", ""),
                    str(failure.get("model_result_decoded"))[:max_field],
                    str(failure.get("possible_answer"))[:max_field],
                    str(failure.get("error"))[:max_field],
                ])
    return buffer.getvalue()


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

        # The actionable view: composite cannot_find_match wrappers unpacked into the
        # sub_error_type they actually hide (see sub_error_types()).
        root = root_cause_counts({group: entries})[group]
        if root != counts[group]:
            root_total = sum(root.values())
            lines.append("")
            lines.append(f"  failures by ROOT cause ({root_total} attributions):")
            for cause, count in root.most_common():
                lines.append(f"    {count:>5}  {count / root_total:>6.1%}  {cause}")

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
