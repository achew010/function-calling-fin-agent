"""Evaluate a locally served checkpoint against the internal, risk-tiered held-out set.

For each conversation, every assistant turn becomes one eval instance: the preceding
turns are sent to the model, and its output is scored against that turn's ground truth
(a function call, or a refusal/clarification). Results are broken out by risk_tier.

Usage:
    python run_internal_eval.py --endpoint http://localhost:8000/v1 --model my-checkpoint \
        --test-file ../0_data/data/test.jsonl --output results/internal_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "0_data"))
from prepare_dataset import try_parse_calls  # noqa: E402

from openai import OpenAI


def load_examples(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def build_eval_instances(example: dict[str, Any]) -> list[dict[str, Any]]:
    """One instance per assistant turn: context up to that turn, plus its ground truth."""
    instances = []
    context: list[dict[str, str]] = [{"role": "system", "content": example["system"]}]
    for turn in example["turns"]:
        if turn["role"] == "assistant":
            instances.append(
                {
                    "context": list(context),
                    "tool_names": {t["name"] for t in example["tools"]},
                    "risk_tier": example["risk_tier"],
                    "expected_calls": try_parse_calls(turn["content"]),
                }
            )
        context.append({"role": turn["role"], "content": turn["content"]})
    return instances


def normalize_calls(calls: list[dict[str, Any]]) -> list[str]:
    """Canonical, order-independent representation for equality comparison.

    Argument values can themselves be dicts/lists (ast.literal_eval doesn't restrict to
    scalars), which aren't orderable — sorting raw (name, args-tuple) pairs raises
    TypeError as soon as two same-named parallel calls need a deeper comparison. Sorting
    JSON strings instead sidesteps that: strings are always comparable.
    """
    return sorted(
        json.dumps(
            {"name": c["name"], "arguments": c.get("arguments", {})},
            sort_keys=True,
            default=str,
        )
        for c in calls
    )


def parse_prediction(text: str) -> list[dict[str, Any]] | None:
    """The fine-tuned model is trained to emit JSON (see 1_training/render_assistant_turn),
    not ToolACE's native call syntax — a non-JSON or non-call response counts as a refusal."""
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(
        isinstance(c, dict) and "name" in c for c in parsed
    ):
        return None
    return parsed


def score_instance(
    instance: dict[str, Any], predicted: list[dict[str, Any]] | None
) -> dict[str, Any]:
    expected = instance["expected_calls"]
    if expected is None:
        return {"is_call_case": False, "refusal_correct": predicted is None}
    if predicted is None:
        return {
            "is_call_case": True,
            "name_correct": False,
            "param_correct": False,
            "hallucinated": False,
        }
    hallucinated = any(c["name"] not in instance["tool_names"] for c in predicted)
    name_correct = {c["name"] for c in predicted} == {c["name"] for c in expected}
    param_correct = normalize_calls(predicted) == normalize_calls(expected)
    return {
        "is_call_case": True,
        "name_correct": name_correct,
        "param_correct": param_correct,
        "hallucinated": hallucinated,
    }


def rate(results: list[dict[str, Any]], key: str) -> float | None:
    if not results:
        return None
    return sum(1 for r in results if r[key]) / len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/internal_eval.json"))
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Qwen3 is a hybrid reasoning model and defaults to non-thinking mode being "
            "explicitly disabled here — leave this off unless you're deliberately "
            "measuring thinking-mode's accuracy/latency tradeoff. Ignored by non-Qwen3 "
            "models."
        ),
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.endpoint, api_key=args.api_key)

    examples = load_examples(args.test_file)
    instances = [inst for ex in examples for inst in build_eval_instances(ex)]
    if args.max_instances:
        instances = instances[: args.max_instances]

    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for inst in instances:
        completion = client.chat.completions.create(
            model=args.model,
            messages=inst["context"],
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": args.enable_thinking}},
        )
        predicted_text = completion.choices[0].message.content or ""
        predicted = parse_prediction(predicted_text)
        by_tier[inst["risk_tier"]].append(score_instance(inst, predicted))

    report: dict[str, Any] = {}
    for tier, results in by_tier.items():
        call_cases = [r for r in results if r["is_call_case"]]
        refusal_cases = [r for r in results if not r["is_call_case"]]
        report[tier] = {
            "n_call_cases": len(call_cases),
            "n_refusal_cases": len(refusal_cases),
            "name_accuracy": rate(call_cases, "name_correct"),
            "param_accuracy": rate(call_cases, "param_correct"),
            "hallucination_rate": rate(call_cases, "hallucinated"),
            "refusal_accuracy": rate(refusal_cases, "refusal_correct"),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
