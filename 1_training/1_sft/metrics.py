"""Function-calling validation metrics, computed from real generations (not
teacher-forced predictions — see FunctionCallEvalCallback in train_sft.py for why that
distinction matters). Reuses 2_evaluations/run_internal_eval.py's parsing/scoring
building blocks rather than reimplementing them, so these numbers mean the same thing
the eval-gate numbers mean.

Metric coverage, reviewed against a broader wishlist request (tool-selection accuracy,
parameter extraction, composed/partial-match, multi-step trajectory, error-type
breakdown):

- Covered directly: tool-selection exact match (handles single/parallel via set
  comparison), per-parameter value/type accuracy, token-level overlap, full-call
  accuracy, the P(tool) / P(params|tool) decomposition, step-level accuracy (= the
  per-instance rate, since one instance already is one turn), full-trajectory accuracy
  for multi-turn conversations, and an error-type breakdown.
- Deliberately NOT included, with reasons:
  - "Top-N accuracy" doesn't map cleanly onto a model that free-generates the call
    rather than classifying over a fixed candidate set via one softmax. The honest
    analogue is pass@N (sample N completions, check if any is correct), which is an
    N-times-more-expensive generation cost — better suited to a dedicated eval run than
    this lightweight per-epoch training callback.
  - "Wrong order" (of multiple calls) isn't modeled: matching is call-name-keyed and
    order-independent by design, mirroring how parallel calls are already compared
    everywhere else in this project (0_data/prepare_dataset.py, run_internal_eval.py).
  - Per-call error typing reports the *first* issue found, in priority order
    (missing param > extra param > wrong type > wrong value), not a full multi-label
    breakdown — a call can have more than one simultaneous issue.
"""

from __future__ import annotations

import json
import re
import random
from collections import Counter
from typing import Any

ERROR_TYPES = [
    "wrong_tool",
    "hallucinated_tool",
    "missed_call",
    "unwarranted_call",
    "missing_param",
    "extra_param",
    "wrong_param_value",
    "wrong_param_type",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def token_overlap_f1(pred_text: str, exp_text: str) -> float:
    """SQuAD-style token-multiset F1 — a smoother, non-binary progress signal than
    exact match, useful early in training when exact-match rates are mostly zero."""
    pred_tokens = Counter(_tokenize(pred_text))
    exp_tokens = Counter(_tokenize(exp_text))
    overlap = sum((pred_tokens & exp_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_tokens.values())
    recall = overlap / sum(exp_tokens.values())
    return 2 * precision * recall / (precision + recall)


def _diff_params(pred_args: dict[str, Any], exp_args: dict[str, Any]) -> str | None:
    """Dominant issue with one call's arguments, or None if they match exactly."""
    if set(exp_args) - set(pred_args):
        return "missing_param"
    if set(pred_args) - set(exp_args):
        return "extra_param"
    for key, exp_val in exp_args.items():
        if type(pred_args.get(key)) is not type(exp_val):
            return "wrong_param_type"
    for key, exp_val in exp_args.items():
        if pred_args.get(key) != exp_val:
            return "wrong_param_value"
    return None


def score_detailed(instance: dict[str, Any], predicted_text: str, parse_prediction, normalize_calls) -> dict[str, Any]:
    """Per-turn detailed scoring. `parse_prediction`/`normalize_calls` are passed in
    (rather than imported at module level) so this module has no import-time
    dependency on run_internal_eval.py's own module-level `openai` import."""
    predicted = parse_prediction(predicted_text)
    expected = instance["expected_calls"]

    if expected is None:
        refusal_correct = predicted is None
        return {
            "is_call_case": False,
            "refusal_correct": refusal_correct,
            "error_type": None if refusal_correct else "unwarranted_call",
        }

    if predicted is None:
        return {
            "is_call_case": True,
            "name_correct": False,
            "param_correct": False,
            "hallucinated": False,
            "param_value_hits": 0,
            "param_value_total": 0,
            "param_type_hits": 0,
            "param_type_total": 0,
            "token_overlap": 0.0,
            "error_type": "missed_call",
        }

    tool_names = set(instance["tool_names"])
    hallucinated = any(c["name"] not in tool_names for c in predicted)
    name_correct = {c["name"] for c in predicted} == {c["name"] for c in expected}
    param_correct = normalize_calls(predicted) == normalize_calls(expected)

    value_hits = type_hits = total_params = 0
    call_error: str | None = None
    expected_names = [c["name"] for c in expected]
    # Pairing predicted-vs-expected calls by name only works when each name appears at
    # most once — a parallel call can invoke the *same* tool multiple times with
    # different arguments (e.g. generating several payment cards in one turn), and a
    # naive {name: args} dict silently collapses those to the last occurrence,
    # mis-pairing the rest. Caught by self-consistency testing this file against real
    # data: a "perfect" prediction scored 83% param_value_accuracy instead of 100%.
    # Skipping the per-parameter breakdown in that case is a correctness trade, not a
    # coverage gap — full_call_accuracy already handles it correctly via set comparison.
    if len(expected_names) == len(set(expected_names)):
        predicted_by_name = {c["name"]: c.get("arguments", {}) for c in predicted}
        for ec in expected:
            exp_args = ec.get("arguments", {})
            pred_args = predicted_by_name.get(ec["name"])
            if pred_args is None:
                continue  # name mismatch — already captured by name_correct/hallucinated
            for key, exp_val in exp_args.items():
                total_params += 1
                if key in pred_args:
                    if type(pred_args[key]) is type(exp_val):
                        type_hits += 1
                    if pred_args[key] == exp_val:
                        value_hits += 1
            if call_error is None:
                call_error = _diff_params(pred_args, exp_args)

    error_type = "hallucinated_tool" if hallucinated else ("wrong_tool" if not name_correct else call_error)

    return {
        "is_call_case": True,
        "name_correct": name_correct,
        "param_correct": param_correct,
        "hallucinated": hallucinated,
        "param_value_hits": value_hits,
        "param_value_total": total_params,
        "param_type_hits": type_hits,
        "param_type_total": total_params,
        "token_overlap": token_overlap_f1(json.dumps(predicted, sort_keys=True), json.dumps(expected, sort_keys=True)),
        "error_type": error_type,
    }


def call_correctness(scored_examples: list[tuple[str, list[dict[str, Any]]]]) -> float | None:
    """Fraction of call-case turns whose full call (name + arguments) exactly matches
    the expected call(s) — a subset of aggregate_metrics' `full_call_accuracy`, split
    out as the single metric SFT logging reports for now (see FunctionCallEvalCallback
    in train_sft.py). Refusal/no-call turns are excluded, matching what
    "call correctness" means as opposed to invocation-decision accuracy."""
    call_results = [r for _, results in scored_examples for r in results if r["is_call_case"]]
    return _rate(call_results, "param_correct")


def _instance_correct(r: dict[str, Any]) -> bool:
    return r.get("param_correct", False) if r["is_call_case"] else r.get("refusal_correct", False)


def _rate(items: list[dict[str, Any]], key: str) -> float | None:
    return sum(1 for i in items if i.get(key)) / len(items) if items else None


def aggregate_metrics(scored_examples: list[tuple[str, list[dict[str, Any]]]]) -> dict[str, float]:
    """`scored_examples`: one (call_type, [per-turn detailed result]) entry per
    conversation — the grouping (not a flat instance list) is what makes trajectory
    accuracy computable."""
    all_results = [r for _, results in scored_examples for r in results]
    n = len(all_results)
    call_results = [r for r in all_results if r["is_call_case"]]
    refusal_results = [r for r in all_results if not r["is_call_case"]]
    name_correct_results = [r for r in call_results if r["name_correct"]]
    error_counts = Counter(r["error_type"] for r in all_results if r.get("error_type"))

    value_total = sum(r.get("param_value_total", 0) for r in call_results)
    value_hits = sum(r.get("param_value_hits", 0) for r in call_results)
    type_total = sum(r.get("param_type_total", 0) for r in call_results)
    type_hits = sum(r.get("param_type_hits", 0) for r in call_results)

    trajectories = [results for call_type, results in scored_examples if call_type == "multi_turn"]
    trajectory_correct = sum(1 for results in trajectories if all(_instance_correct(r) for r in results))

    metrics: dict[str, float | None] = {
        "n_conversations": len(scored_examples),
        "n_trajectories": len(trajectories),
        "n_instances": n,
        "n_call_cases": len(call_results),
        "n_refusal_cases": len(refusal_results),
        "tool_selection_exact_match": _rate(call_results, "name_correct"),
        "full_call_accuracy": _rate(call_results, "param_correct"),
        "param_accuracy_given_tool_correct": _rate(name_correct_results, "param_correct"),
        "param_value_accuracy": (value_hits / value_total) if value_total else None,
        "param_type_accuracy": (type_hits / type_total) if type_total else None,
        "token_overlap_f1": (
            sum(r.get("token_overlap", 0.0) for r in call_results) / len(call_results) if call_results else None
        ),
        "hallucination_rate": _rate(call_results, "hallucinated"),
        "refusal_accuracy": _rate(refusal_results, "refusal_correct"),
        "trajectory_accuracy": (trajectory_correct / len(trajectories)) if trajectories else None,
    }
    for error_type in ERROR_TYPES:
        metrics[f"error_rate_{error_type}"] = (error_counts.get(error_type, 0) / n) if n else None

    decision_rates = [metrics[key] for key in ("full_call_accuracy", "refusal_accuracy") if metrics[key] is not None]
    metrics["balanced_call_accuracy"] = sum(decision_rates) / len(decision_rates) if decision_rates else None

    return {k: v for k, v in metrics.items() if v is not None}


def summarize_error_gaps(conversations: list[dict], parse_prediction) -> dict:
    """Overlapping diagnostic slices; flags are not semantic root-cause labels.

    Save IDs for inspection, and denominators so tiny slices stay visible. In
    particular, a prerequisite keyword is a proxy, not proof of premature action.
    """
    slices = {}
    for conversation in conversations:
        for turn in conversation["turns"]:
            score = turn["score"]
            calls = turn["expected_calls"]
            keys = [conversation["call_type"], "call" if calls is not None else "no_call"]
            context = turn["context"]
            if context and (context[-1]["role"] == "tool" or re.search(
                r"\b(before|after|once|first|then|if)\b", context[-1]["content"], re.I
            )):
                keys.append("prerequisite_proxy")
            if calls and len({c["name"] for c in calls}) < len(calls):
                keys.append("repeated_tool_calls")
            predicted = parse_prediction(turn["prediction"]) or []
            for key in set(keys):
                row = slices.setdefault(key, {"n": 0, "correct": 0, "truncated": 0,
                                              "wrong_call_count": 0, "failures": []})
                row["n"] += 1
                row["correct"] += int(_instance_correct(score))
                row["truncated"] += int(turn["reached_token_limit"])
                row["wrong_call_count"] += int(calls is not None and len(predicted) != len(calls))
                if not _instance_correct(score):
                    row["failures"].append({"conversation_id": conversation["conversation_id"],
                                            "turn_index": turn["turn_index"],
                                            "error_type": score.get("error_type")})
    for row in slices.values():
        row["accuracy"] = row["correct"] / row["n"]
    return slices


def compare_validation_reports(baseline: dict, candidate: dict, n_bootstrap: int = 10000, seed: int = 0) -> dict:
    """Paired percentile bootstrap of candidate-minus-baseline accuracy.

    Resample conversations with replacement, retaining all their turns and pairing
    identical IDs in both runs. Call/refusal accuracy remains turn-weighted; trajectory
    accuracy remains conversation-weighted. These are pointwise, not selection-adjusted,
    intervals. A replicate with no eligible denominator is omitted for that metric.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    for key in ("schema_version", "dataset_sha256", "scorer_sha256", "generation"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Incompatible validation reports: {key}")

    def index(report):
        rows = report["conversations"]
        indexed = {row["conversation_id"]: row for row in rows}
        if not rows or len(indexed) != len(rows):
            raise ValueError("Conversation IDs must be nonempty and unique")
        return indexed

    left, right = index(baseline), index(candidate)
    if left.keys() != right.keys():
        raise ValueError("Validation conversation IDs differ")
    stats = []
    for cid in sorted(left):
        a, b = left[cid], right[cid]
        if a["call_type"] != b["call_type"] or len(a["turns"]) != len(b["turns"]):
            raise ValueError(f"Conversation structure differs: {cid}")
        for x, y in zip(a["turns"], b["turns"]):
            for key in ("turn_index", "expected_calls", "context"):
                if x[key] != y[key]:
                    raise ValueError(f"Turn alignment differs: {cid}/{key}")
            if x["score"]["is_call_case"] != y["score"]["is_call_case"]:
                raise ValueError(f"Call-case labels differ: {cid}")

        def counts(row):
            scores = [t["score"] for t in row["turns"]]
            calls = [s for s in scores if s["is_call_case"]]
            refusals = [s for s in scores if not s["is_call_case"]]
            trajectory = row["call_type"] == "multi_turn"
            return [
                (sum(s.get("param_correct", False) for s in calls), len(calls)),
                (sum(s.get("refusal_correct", False) for s in refusals), len(refusals)),
                (int(trajectory and bool(scores) and all(_instance_correct(s) for s in scores)), int(trajectory)),
            ]
        stats.append((counts(a), counts(b)))

    names = ["full_call_accuracy", "refusal_accuracy", "trajectory_accuracy"]
    def rates(indices, metric):
        an = ad = bn = bd = 0
        for i in indices:
            a, b = stats[i]
            an += a[metric][0]
            ad += a[metric][1]
            bn += b[metric][0]
            bd += b[metric][1]
        return (an / ad, bn / bd) if ad and bd else None

    draws = [[] for _ in names]
    rng = random.Random(seed)
    for _ in range(n_bootstrap):
        indices = rng.choices(range(len(stats)), k=len(stats))
        for m in range(len(names)):
            pair = rates(indices, m)
            if pair is not None:
                draws[m].append(pair[1] - pair[0])

    def quantile(values, q):
        pos = (len(values) - 1) * q
        low = int(pos)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (pos - low)

    result = {"baseline_step": baseline["step"], "candidate_step": candidate["step"],
              "n_conversations": len(stats), "n_bootstrap": n_bootstrap, "seed": seed,
              "method": "paired conversation-cluster percentile bootstrap; pointwise 95% CI",
              "metrics": {}}
    for m, name in enumerate(names):
        pair = rates(range(len(stats)), m)
        if pair is not None and draws[m]:
            values = sorted(draws[m])
            result["metrics"][name] = {
                "baseline": pair[0], "candidate": pair[1], "delta": pair[1] - pair[0],
                "ci_low": quantile(values, 0.025), "ci_high": quantile(values, 0.975),
                "valid_replicates": len(values),
            }
    return result
