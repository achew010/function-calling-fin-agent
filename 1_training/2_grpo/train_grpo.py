"""GRPO fine-tune of a function-calling policy, via TRL's GRPOTrainer.

Starting point: the SFT checkpoint from ../1_sft/ (see that README for why RL needs a
warm-started policy rather than the raw base model — GRPO gets little useful reward
signal from a policy that rarely produces a valid completion to begin with).

Reward: mirrors 2_evaluations/run_internal_eval.py's scoring dimensions (function-name
match, parameter match, hallucination penalty, refusal correctness), collapsed into one
scalar with typed argument partial credit and one-to-one matching of repeated calls.
Exact-call matching reuses the evaluator's canonicalization; the shaped reward is a
training surrogate, not the checkpoint-selection metric. --reward-mode legacy restores
the original flat partial credit for a controlled ablation. See the README for limits
of parser-based refusal scoring and the risk-tier-weighting extension.

Dataset: reuses 2_evaluations/run_internal_eval.py's build_eval_instances (imported, not
reimplemented) to turn each conversation into one training prompt per assistant turn —
the exact same instances the internal eval scores checkpoints against.

Usage:
    python train_grpo.py \
        --base-model ../1_sft/checkpoints/sft-general \
        --train-file ../../0_data/data/train.jsonl \
        --val-file ../../0_data/data/val.jsonl \
        --output-dir checkpoints/grpo-general
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import random
import re
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

# Reuse the exact instance-construction and scoring logic 2_evaluations/run_internal_eval.py
# uses, rather than re-implementing it — the reward function and the eval gate should stay
# one piece of logic, not two that can quietly drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "2_evaluations"))
from run_internal_eval import build_eval_instances, load_examples, normalize_calls, parse_prediction  # noqa: E402

# Reuse the exact generation-based eval callback train_sft.py already built (real
# model.generate() against a held-out sample, scored via metrics.py's aggregate_metrics)
# rather than reimplementing it a second time -- same reasoning as reusing
# run_internal_eval.py above: the eval mechanism and the metric name it reports
# (fc_call_correctness) should be identical across both stages, not two copies that can
# quietly drift apart. Needs train_sft.py + metrics.py present alongside this file --
# see grpo-job.yaml/grpo-smoke-job.yaml's ConfigMap build commands.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_sft"))
from train_sft import FunctionCallEvalCallback  # noqa: E402

import statistics

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from scipy.optimize import linear_sum_assignment


def build_grpo_dataset(path: Path, example_ids: list[int] | None = None) -> Dataset:
    """One row per assistant turn (same instances run_internal_eval.py scores), with the
    ground truth needed to compute a reward carried alongside the prompt rather than
    baked into a single target completion.

    `expected_calls` is stored JSON-encoded rather than as a nested column: argument
    values are heterogeneous (str/int/float/list/dict) across examples, which Arrow's
    struct-column type inference can't represent — `Dataset.from_dict` raises
    `ArrowInvalid: cannot mix struct and non-struct, non-null values` on the raw nested
    form. reward_func decodes it back with json.loads per-row.
    """
    examples = load_examples(path)
    if example_ids is not None:
        examples = [examples[i] for i in example_ids]
    prompts, tool_names_col, is_call_case_col, expected_calls_col = [], [], [], []
    for ex in examples:
        for instance in build_eval_instances(ex):
            prompts.append(instance["context"])
            tool_names_col.append(sorted(instance["tool_names"]))
            is_call_case_col.append(instance["expected_calls"] is not None)
            expected_calls_col.append(json.dumps(instance["expected_calls"] or []))
    return Dataset.from_dict(
        {
            "prompt": prompts,
            "tool_names": tool_names_col,
            "is_call_case": is_call_case_col,
            "expected_calls": expected_calls_col,
        }
    )


def context_hash(context: list[dict]) -> str:
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def grounded_arguments(calls: list[dict], context: list[dict]) -> bool:
    """Conservative pilot filter, not a semantic verifier.

    Every scalar argument must appear in user/tool context. Exclude assistant prose
    and schemas so invented earlier answers and example/default values don't qualify.
    This deliberately drops valid examples needing inference or canonicalization.
    """
    text = "\n".join(m["content"] for m in context if m["role"] in {"user", "tool"}).casefold()
    def supported(value):
        if isinstance(value, dict):
            return all(supported(v) for v in value.values())
        if isinstance(value, list):
            return all(supported(v) for v in value)
        if type(value) in {int, float}:
            return any(float(n) == value for n in re.findall(r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)", text))
        if isinstance(value, str):
            return bool(value.strip()) and value.casefold() in text
        return str(value).casefold() in text
    return all(supported(call.get("arguments", {})) for call in calls)


def build_pilot_dataset(train_file: Path, val_file: Path, size: int = 256, seed: int = 42):
    """Select one target per training conversation, half no-call and half valid-call.

    No-call targets must follow a user and contain an explicit clarification or
    inability statement in the reference. Call targets require grounded arguments;
    prerequisite-related cases are prioritized. Exact validation-context overlaps
    exclude the entire training conversation. No validation predictions guide mining.
    """
    if size < 2 or size % 2:
        raise ValueError("Pilot size must be a positive even number of at least 2")
    excluded = {context_hash(inst["context"]) for ex in load_examples(val_file)
                for inst in build_eval_instances(ex)}
    buckets = {"clarify_or_abstain": [], "prerequisite_call": [], "grounded_call": []}
    seen = set()
    for cid, ex in enumerate(load_examples(train_file)):
        instances = build_eval_instances(ex)
        if any(context_hash(inst["context"]) in excluded for inst in instances):
            continue
        references = [turn["content"] for turn in ex["turns"] if turn["role"] == "assistant"]
        candidates = []
        for tid, (inst, reference) in enumerate(zip(instances, references)):
            context, calls = inst["context"], inst["expected_calls"]
            if not context:
                continue
            if calls is None:
                if context[-1]["role"] != "user" or not re.search(
                    r"\b(provide|specify|clarify|missing|required|cannot|can't|unable|not available|need)\b",
                    reference, re.IGNORECASE,
                ):
                    continue
                bucket = "clarify_or_abstain"
            elif calls and grounded_arguments(calls, context):
                prerequisite = context[-1]["role"] == "tool" or re.search(
                    r"\b(before|after|once|first|then|if)\b", context[-1]["content"], re.IGNORECASE)
                bucket = "prerequisite_call" if prerequisite else "grounded_call"
            else:
                continue
            candidates.append((bucket, tid, inst, reference))
        # Prefer clarification targets, then prerequisite calls; at most one per conversation.
        candidates.sort(key=lambda item: list(buckets).index(item[0]))
        for bucket, tid, inst, reference in candidates:
            fingerprint = context_hash(inst["context"])
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            buckets[bucket].append({
                "prompt": inst["context"], "tool_names": sorted(inst["tool_names"]),
                "is_call_case": inst["expected_calls"] is not None,
                "expected_calls": json.dumps(inst["expected_calls"] or []),
                "source_conversation_id": cid, "source_turn_index": tid,
                "pilot_bucket": bucket, "reference": reference,
            })
            break
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    half = size // 2
    calls = buckets["prerequisite_call"] + buckets["grounded_call"]
    if len(buckets["clarify_or_abstain"]) < half or len(calls) < half:
        raise ValueError(f"Not enough eligible pilot examples for {size}: "
                         f"{ {key: len(value) for key, value in buckets.items()} }")
    # Reserve half the call examples for ordinary valid calls, with backfill if
    # either pool is small. Prerequisite cases must not crowd out the guard set.
    quota = half // 2
    chosen_calls = buckets["prerequisite_call"][:quota] + buckets["grounded_call"][:half - quota]
    chosen_ids = {row["source_conversation_id"] for row in chosen_calls}
    chosen_calls += [row for row in calls if row["source_conversation_id"] not in chosen_ids][:half - len(chosen_calls)]
    selected = buckets["clarify_or_abstain"][:half] + chosen_calls
    rng.shuffle(selected)
    manifest = {"seed": seed, "n_targets": len(selected),
                "eligible_counts": {key: len(value) for key, value in buckets.items()},
                "selected_counts": dict(Counter(row["pilot_bucket"] for row in selected)),
                "train_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
                "val_sha256": hashlib.sha256(val_file.read_bytes()).hexdigest(),
                "examples": selected}
    columns = ("prompt", "tool_names", "is_call_case", "expected_calls")
    return Dataset.from_list([{key: row[key] for key in columns} for row in selected]), manifest


class StallStopCallback(TrainerCallback):
    """Abort a run that hasn't moved at all by `check_step`.

    Deliberately NOT transformers' EarlyStoppingCallback, which stops once a metric
    stops IMPROVING after some patience. This answers a different question: by step N,
    has the run moved *at all*? A GRPO run whose prompt groups are mostly reward-tied
    produces near-zero advantage and therefore near-zero gradient, and its greedy
    validation metrics then sit exactly where they started -- measured at 50-85% of
    groups tied. That run is worth killing at step 25 rather than letting it spend its
    full budget proving the same point.

    Reads the metrics dict rather than the logs because FunctionCallEvalCallback mutates
    that dict in place (train_sft.py's `metrics.update(...)` / `metrics[...] = ...`), so
    the generation-based numbers are visible here -- but ONLY if this callback is
    ordered after it in GRPOTrainer(callbacks=[...]).

    Baseline comes from the step-0 evaluation, which exists because GRPOConfig sets
    eval_on_start=True.
    """

    def __init__(self, metric_names: list[str], check_step: int, min_delta: float) -> None:
        self.metric_names = list(metric_names)
        self.check_step = check_step
        self.min_delta = min_delta
        self.baseline: dict[str, float] = {}
        self.warned_missing = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        observed = {name: metrics[name] for name in self.metric_names if name in metrics}
        missing = [name for name in self.metric_names if name not in metrics]
        if missing and not self.warned_missing:
            self.warned_missing = True
            # Loud once: a typo'd metric name would otherwise mean this callback
            # silently never fires, which looks identical to "the run is progressing".
            print(f"WARNING: StallStopCallback cannot see {missing}; watching {sorted(observed)} only. "
                  f"Available: {sorted(k for k in metrics if k.startswith('eval_'))}", flush=True)
        if not observed:
            return
        if not self.baseline:
            self.baseline = observed
            print(f"[stall-check] baseline at step {state.global_step}: "
                  f"{ {k: round(v, 4) for k, v in observed.items()} }", flush=True)
            return
        if state.global_step < self.check_step:
            return
        deltas = {name: abs(value - self.baseline[name])
                  for name, value in observed.items() if name in self.baseline}
        if deltas and max(deltas.values()) < self.min_delta:
            print(f"[stall-check] STOPPING at step {state.global_step}: no watched metric moved by "
                  f"{self.min_delta} since baseline ({ {k: round(v, 6) for k, v in deltas.items()} }). "
                  f"The run is not learning -- check the tied-group fraction in pilot_selection.json "
                  f"and the effective learning rate before spending more budget.", flush=True)
            control.should_training_stop = True


@torch.no_grad()
def rollout_reward_groups(
    rows: list[dict[str, Any]], base_model: str, num_generations: int, temperature: float,
    max_new_tokens: int, batch_size: int, reward_mode: str,
) -> list[list[float]]:
    """Sample `num_generations` completions per row from the CURRENT policy and score
    each with compute_reward -- i.e. the exact group of rewards GRPO would compute for
    that prompt on step 1.

    Loads and frees its own copy of the model: GRPOTrainer takes a model *path* and
    loads internally, so there is nothing to borrow at selection time. The two loads are
    sequential, not concurrent, so peak memory is unchanged.

    Sampling deliberately mirrors the training rollouts (same temperature, same
    enable_thinking=False chat template as grpo_config.chat_template_kwargs) -- a group
    scored under different sampling than training sees would be measuring the wrong
    distribution.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    padding_side = tokenizer.padding_side
    # torch_dtype/device_map spelled exactly as train_sft.py loads the same checkpoint --
    # the kwarg name changed across transformers releases, so match the call this repo
    # already runs against its pinned version rather than the newer spelling.
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype="bfloat16")
    eos_ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
    groups: list[list[float]] = []
    try:
        model.eval()
        tokenizer.padding_side = "left"  # decoder-only batched generation
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset: offset + batch_size]
            print(f"[select] rollouts {offset + 1}-{offset + len(batch)}/{len(rows)}", flush=True)
            prompts = [
                tokenizer.apply_chat_template(
                    row["prompt"], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                for row in batch
            ]
            inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_generations,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids or None,
            )
            # generate() returns rows grouped per input: [row0 x G, row1 x G, ...].
            prompt_length = inputs["input_ids"].shape[1]
            for index, row in enumerate(batch):
                expected_calls = json.loads(row["expected_calls"])
                rewards = []
                for sequence in outputs[index * num_generations: (index + 1) * num_generations]:
                    completion_ids = sequence[prompt_length:].tolist()
                    # Batched completions are padded out to the longest in the batch;
                    # count only through the first stopping token.
                    stop = next((i for i, token in enumerate(completion_ids) if token in eos_ids), None)
                    if stop is not None:
                        completion_ids = completion_ids[: stop + 1]
                    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                    rewards.append(compute_reward(
                        text, row["tool_names"], row["is_call_case"], expected_calls, reward_mode))
                groups.append(rewards)
    finally:
        tokenizer.padding_side = padding_side
        del model
        torch.cuda.empty_cache()
    return groups


def select_informative_rows(
    rows: list[dict[str, Any]], groups: list[list[float]], target_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep only prompts whose rollouts DISAGREE, highest-spread first, bucket-balanced.

    This is the whole point of rollout-based selection. GRPO's advantage is
    (reward - group_mean) / group_std, so a group where every rollout earns the same
    reward contributes exactly zero advantage to every one of its members and therefore
    zero gradient -- whether the model got it always right or always wrong. Selecting by
    heuristic category (build_pilot_dataset's buckets) says nothing about whether the
    CURRENT policy is uncertain there, which is why most groups came back tied.

    Bucket balance is preserved by round-robin across buckets rather than taking a flat
    top-N: spread alone would happily return an all-abstention set and quietly drop the
    call-side guard examples build_pilot_dataset deliberately reserves.
    """
    scored: dict[str, list[tuple[float, dict[str, Any]]]] = {}
    zero_variance = 0
    for row, rewards in zip(rows, groups):
        spread = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        if spread == 0.0:  # always-correct or always-wrong: no learning signal
            zero_variance += 1
            continue
        scored.setdefault(row.get("pilot_bucket", "unbucketed"), []).append((spread, row))

    for candidates in scored.values():
        candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[dict[str, Any]] = []
    while len(selected) < target_size and any(scored.values()):
        for candidates in scored.values():
            if candidates and len(selected) < target_size:
                spread, row = candidates.pop(0)
                selected.append({**row, "rollout_reward_spread": spread})

    informative = sum(len(v) for v in scored.values()) + len(selected)
    stats = {
        "candidates": len(rows),
        "zero_variance_dropped": zero_variance,
        "informative": informative,
        "informative_fraction": round(informative / len(rows), 4) if rows else 0.0,
        "selected": len(selected),
        "selected_counts": dict(Counter(row.get("pilot_bucket", "unbucketed") for row in selected)),
        "mean_selected_spread": round(
            statistics.fmean(row["rollout_reward_spread"] for row in selected), 4) if selected else 0.0,
    }
    return selected, stats


def select_probe_ids(val_file: Path, size: int = 128, seed: int = 42) -> list[int]:
    """Round-robin stratification by conversation call_type; never by model errors."""
    if size < 1:
        raise ValueError("Probe size must be positive")
    buckets = {}
    for cid, ex in enumerate(load_examples(val_file)):
        buckets.setdefault(ex["call_type"], []).append(cid)
    rng = random.Random(seed)
    for rows in buckets.values():
        rng.shuffle(rows)
    chosen = []
    while len(chosen) < size and any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key] and len(chosen) < size:
                chosen.append(buckets[key].pop())
    return sorted(chosen)


def apply_pilot_defaults(args, argv):
    """Explicit CLI overrides win over the pilot preset."""
    if not args.pilot:
        return
    defaults = {"max_steps": 100, "num_generations": 4, "grad_accum": 4,
                "eval_steps": 25, "save_steps": 25, "logging_steps": 5,
                "max_completion_length": 512, "metrics_max_new_tokens": 512}
    for key, value in defaults.items():
        flag = "--" + key.replace("_", "-")
        if not any(arg == flag or arg.startswith(flag + "=") for arg in argv):
            setattr(args, key, value)


def typed_equal(left: Any, right: Any) -> bool:
    """Exact recursive equality: booleans, integers and floats are distinct."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(typed_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def argument_credit(predicted: dict, expected: dict) -> float:
    """Union of keys penalizes both omitted and invented arguments.

    A present argument earns 1/4 for its top-level type and 3/4 for exact typed value.
    """
    keys = predicted.keys() | expected.keys()
    if not keys:
        return 1.0
    return sum(
        0.25 * (type(predicted[k]) is type(expected[k])) + 0.75 * typed_equal(predicted[k], expected[k])
        for k in keys if k in predicted and k in expected
    ) / len(keys)


def compute_reward(
    text: str, tool_names: list[str], is_call_case: bool, expected_calls: list[dict[str, Any]],
    reward_mode: str = "dense",
) -> float:
    if reward_mode not in {"dense", "legacy"}:
        raise ValueError(f"Unknown reward mode: {reward_mode}")
    if reward_mode == "dense" and not text.strip():
        return -1.0  # an empty generation is not a successful refusal
    predicted = parse_prediction(text)
    if not is_call_case:
        return 1.0 if predicted is None else -1.0  # correct refusal vs. an unwarranted call
    if predicted is None:
        return -1.0  # missed a call it should have made
    if any(c["name"] not in tool_names for c in predicted):
        return -2.0  # hallucinated a call — worst case, matches the risk-tiering framing
    if {c["name"] for c in predicted} != {c["name"] for c in expected_calls}:
        return -0.5  # wrong function entirely
    if reward_mode == "legacy":
        return 1.0 if normalize_calls(predicted) == normalize_calls(expected_calls) else 0.3
    if Counter(c["name"] for c in predicted) != Counter(c["name"] for c in expected_calls):
        return -0.5  # missing/extra invocations, including duplicates of a known tool
    if normalize_calls(predicted) == normalize_calls(expected_calls):
        return 1.0  # preserve a distinct exact-call bonus

    total_credit = 0.0
    for name in sorted({c["name"] for c in expected_calls}):
        actual = [c.get("arguments", {}) for c in predicted if c["name"] == name]
        target = [c.get("arguments", {}) for c in expected_calls if c["name"] == name]
        credits = [[argument_credit(p, e) for e in target] for p in actual]
        # One-to-one maximum-weight matching: a good prediction cannot earn credit
        # for several expected invocations of the same tool.
        rows, columns = linear_sum_assignment(credits, maximize=True)
        total_credit += sum(credits[i][j] for i, j in zip(rows, columns))
    return 0.1 + 0.7 * total_credit / len(expected_calls)


def reward_func(
    completions: list[list[dict[str, str]]],
    tool_names: list[list[str]],
    is_call_case: list[bool],
    expected_calls: list[str],
    reward_mode: str = "dense",
    **kwargs: Any,
) -> list[float]:
    """GRPOTrainer's reward-function contract: `completions` is one message per
    generation (`[{"content": "..."}]`); every other training-dataset column is passed
    through as a same-length list, aligned to `completions` (verified against the
    installed trl version's GRPOTrainer/trl.rewards source — not assumed). `expected_calls`
    arrives JSON-encoded (see build_grpo_dataset) and is decoded here."""
    if len({len(completions), len(tool_names), len(is_call_case), len(expected_calls)}) != 1:
        raise ValueError("Reward inputs must have matching lengths")
    texts = [c[0].get("content") or "" for c in completions]
    return [
        compute_reward(text, names, is_case, json.loads(calls_json), reward_mode=reward_mode)
        for text, names, is_case, calls_json in zip(texts, tool_names, is_call_case, expected_calls)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-model",
        default="../1_sft/checkpoints/sft-general",
        help="Starting policy — the SFT checkpoint from ../1_sft/, not the raw base model (see module docstring).",
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument(
        "--val-file",
        type=Path,
        required=True,
        help="Held-out split for FunctionCallEvalCallback (see train_sft.py) -- GRPO "
        "previously had no held-out eval at all, just the training-rollout reward, so "
        "there was nothing to catch a reward-optimized policy drifting away from real "
        "correctness, and no 'best' checkpoint to restore if it did.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true", help="Targeted 100-update run with balanced training targets and a stratified validation probe.")
    parser.add_argument("--pilot-train-samples", type=int, default=256)
    parser.add_argument("--pilot-eval-samples", type=int, default=128)
    parser.add_argument("--pilot-seed", type=int, default=42)
    parser.add_argument(
        "--select-by-rollout",
        action="store_true",
        help="Select training prompts by actual post-SFT rollout DISAGREEMENT instead of "
        "by heuristic bucket alone. Rolls out --select-rollouts completions per pilot "
        "candidate with the base checkpoint, scores each with the training reward, and "
        "keeps only prompts whose rewards differ within the group. Groups whose rollouts "
        "all score the same produce zero GRPO advantage and therefore zero gradient, so "
        "spending updates on them is the dominant waste in a short run -- measured at "
        "50-85% of groups tied (~70% average), i.e. roughly one informative prompt per "
        "update at 4 unique prompts per update.",
    )
    parser.add_argument("--select-rollouts", type=int, default=8, help="Completions sampled per candidate during selection.")
    parser.add_argument("--select-target-size", type=int, default=96, help="How many informative prompts to keep for training.")
    parser.add_argument(
        "--select-max-new-tokens",
        type=int,
        default=256,
        help="Generation cap during selection only. Real call outputs run tens of tokens, "
        "so the training-time 512 mostly buys worst-case headroom the selection pass "
        "doesn't need -- and selection cost is linear in this.",
    )
    parser.add_argument("--select-batch-size", type=int, default=8, help="Candidates per generation batch (each expands by --select-rollouts).")
    parser.add_argument(
        "--stall-check-step",
        type=int,
        default=0,
        help="Stop the run at this step if NO watched metric has moved since the step-0 "
        "baseline (0 disables). Not 'stopped improving' -- 'never moved', the signature "
        "of a run whose reward groups are tied and whose gradient is therefore ~0. "
        "Pairs with --eval-steps: the check can only fire on an evaluation step.",
    )
    parser.add_argument(
        "--stall-metrics",
        nargs="+",
        default=["eval_fc_call_correctness", "eval_refusal_accuracy"],
        help="Metrics --stall-check-step watches. Must be names FunctionCallEvalCallback "
        "puts in the metrics dict (eval_-prefixed); a name it can't see is warned about "
        "once rather than silently ignored.",
    )
    parser.add_argument(
        "--stall-min-delta",
        type=float,
        default=1e-4,
        help="Movement below this counts as no movement. Small but nonzero: a stalled "
        "GRPO run's greedy predictions are usually bit-identical, so the real deltas are "
        "exactly 0.0 and this only has to clear float noise.",
    )
    parser.add_argument("--num-generations", type=int, default=8, help="Group size G — completions sampled per prompt.")
    parser.add_argument(
        "--num-generations-eval",
        type=int,
        default=2,
        help="Group size for the held-out eval loop, kept far below --num-generations (8) "
        "on purpose: eval only needs completions to score, not a group wide enough to "
        "estimate advantages from, so 8 rollouts per eval prompt is 4x the generation "
        "cost for no extra signal. Also load-bearing for startup: GRPOConfig rejects "
        "any config where per_device_eval_batch_size * world_size isn't divisible by "
        "this (verified against the installed trl's own __post_init__, not assumed) — "
        "at the previous default (None -> falls back to num_generations=8) against "
        "--eval-batch-size 4, that check failed and the run died before training began.",
    )
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reward-mode", choices=["dense", "legacy"], default="dense",
                        help="Dense typed argument credit (default), or the original flat 0.3 partial reward for ablations.")
    parser.add_argument(
        "--kl-beta",
        type=float,
        default=0.0,
        help="KL penalty against the reference policy. TRL's current default (0.0, DAPO-style) "
        "relies on loss clipping alone; set e.g. 0.04 for classic GRPO-style KL regularization "
        "if the policy drifts too far from the SFT checkpoint.",
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument(
        "--lr-scheduler-type",
        default="linear",
        help="transformers' scheduler name. The default 'linear' decays --lr to zero over "
        "the run, so a 50-100 update run spends most of its updates at well under half the "
        "nominal rate (1e-6 nominal -> ~5e-7 average). Use 'constant_with_warmup' with "
        "--warmup-steps for a short run where the decay tail buys nothing.",
    )
    parser.add_argument("--warmup-steps", type=int, default=0, help="Linear warmup updates before the scheduler above takes over.")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Overrides --epochs when > 0 — e.g. for a smoke run of a fixed number of steps regardless of dataset size.",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=4,
        help="4 (was 8): halves the activations held per step. Cannot be "
        "lowered on its own -- TRL derives generation_batch_size from "
        "per_device_batch_size * world_size * --grad-accum and rejects any value not "
        "divisible by --num-generations, since a generation batch has to hold whole "
        "prompt groups. 4 with --grad-accum 2 keeps generation_batch_size at 8 (= G), "
        "so this halves peak memory without changing the effective batch or group size; "
        "dropping to 4 with --grad-accum 1 fails at startup instead.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="Paired with --per-device-batch-size above to keep "
        "per_device_batch_size * --grad-accum a multiple of --num-generations. Also "
        "sets the default steps_per_generation, measured in forward/backward "
        "microsteps, not optimizer updates.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="Decoupled from --per-device-batch-size, same reasoning as train_sft.py's "
        "own --eval-batch-size: eval's own forward/generate passes stack on top of "
        "whatever training already has resident, so it gets its own, smaller knob "
        "rather than silently scaling with the train batch size.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--eval-strategy",
        default="steps",
        choices=["no", "steps", "epoch"],
        help="'steps' (not 'epoch') by default, matching train_sft.py -- fires the "
        "held-out eval at a fixed cadence regardless of dataset size or epoch count. "
        "Paired with --eval-steps 140 below.",
    )
    parser.add_argument("--eval-steps", type=int, default=140)
    parser.add_argument(
        "--save-strategy",
        default="steps",
        choices=["no", "steps", "epoch"],
        help="'steps' (not the previous default 'epoch'): load_best_model_at_end "
        "requires save_strategy == eval_strategy, with save_steps a multiple of "
        "eval_steps (see train_sft.py's identical requirement) -- 'epoch' can't "
        "satisfy that whenever --max-steps stops training before an epoch completes.",
    )
    parser.add_argument("--save-steps", type=int, default=140)
    parser.add_argument(
        "--metrics-eval-samples",
        type=int,
        default=0,
        help="Validation conversations for generation metrics: 0 uses the full split; positive values select a fixed smoke-test subset.",
    )
    parser.add_argument("--metrics-max-new-tokens", type=int, default=256)
    parser.add_argument("--metrics-batch-size", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--mlflow", action="store_true", help="Log metrics to MLflow via transformers' built-in MLflowCallback."
    )
    parser.add_argument("--mlflow-experiment-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None, help="Defaults to local ./mlruns if unset.")
    args = parser.parse_args()
    apply_pilot_defaults(args, sys.argv[1:])

    probe_ids = None
    if args.pilot:
        train_dataset, selection = build_pilot_dataset(
            args.train_file, args.val_file, size=args.pilot_train_samples, seed=args.pilot_seed)
        if args.select_by_rollout:
            # build_pilot_dataset's output is now the CANDIDATE pool, not the training
            # set: it selects by heuristic bucket, which says nothing about whether this
            # checkpoint is actually uncertain on a prompt. Narrow it to the prompts
            # whose rollouts disagree -- see select_informative_rows' docstring.
            candidates = selection["examples"]
            print(f"GRPO selection: rolling out {args.select_rollouts} completions for "
                  f"{len(candidates)} candidates ...", flush=True)
            groups = rollout_reward_groups(
                candidates, args.base_model, num_generations=args.select_rollouts,
                temperature=args.temperature, max_new_tokens=args.select_max_new_tokens,
                batch_size=args.select_batch_size, reward_mode=args.reward_mode)
            chosen, stats = select_informative_rows(candidates, groups, args.select_target_size)
            if not chosen:
                raise ValueError(
                    "Rollout selection kept no prompts: every candidate group scored "
                    "identically across rollouts. Raise --select-rollouts or --temperature, "
                    "or widen --pilot-train-samples.")
            selection["examples"] = chosen
            selection["rollout_selection"] = {
                "rollouts_per_candidate": args.select_rollouts,
                "temperature": args.temperature,
                "max_new_tokens": args.select_max_new_tokens,
                "reward_mode": args.reward_mode,
                "base_model": str(args.base_model),
                **stats,
            }
            selection["selected_counts"] = stats["selected_counts"]
            selection["n_targets"] = len(chosen)
            columns = ("prompt", "tool_names", "is_call_case", "expected_calls")
            train_dataset = Dataset.from_list([{key: row[key] for key in columns} for row in chosen])
            print(f"GRPO selection: {stats['informative']}/{stats['candidates']} candidates informative "
                  f"({stats['informative_fraction']:.0%}); {stats['zero_variance_dropped']} tied groups dropped; "
                  f"training on {stats['selected']} ({stats['selected_counts']})", flush=True)
        probe_ids = select_probe_ids(args.val_file, args.pilot_eval_samples, args.pilot_seed)
        selection["validation_conversation_ids"] = probe_ids
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "pilot_selection.json").write_text(json.dumps(selection, indent=2))
        print(f"GRPO pilot: {selection['selected_counts']}; {len(probe_ids)} validation conversations; "
              f"{args.max_steps} updates", flush=True)
    else:
        train_dataset = build_grpo_dataset(args.train_file)
    # Also gives GRPOTrainer's own built-in eval loop (reward/kl/entropy/completion-
    # length, computed the same way as training rollouts) on real held-out data for
    # free, on top of FunctionCallEvalCallback's richer metrics.py breakdown below --
    # both need a non-None eval_dataset to fire at all (Trainer raises otherwise).
    eval_dataset = build_grpo_dataset(args.val_file, example_ids=probe_ids)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=str(args.output_dir),
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.kl_beta,
        # Qwen3 is a hybrid reasoning model — same non-thinking requirement as everywhere
        # else in this project (2_evaluations/README.md), so rollouts during training
        # match the behavior the model is actually served with.
        chat_template_kwargs={"enable_thinking": False},
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        eval_on_start=True,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        # Mirrors train_sft.py: reload the checkpoint with the best real,
        # generation-based fc_call_correctness (FunctionCallEvalCallback below) at the
        # end of train(), rather than merging/saving whatever the final step produced --
        # this is the actual fix for GRPO previously having no protection against a
        # reward-optimized policy drifting away from real correctness over the run.
        load_best_model_at_end=True,
        metric_for_best_model="balanced_call_accuracy" if args.pilot else "fc_call_correctness",
        greater_is_better=True,
        report_to=[],
    )

    fc_callback = FunctionCallEvalCallback(
        args.val_file,
        eval_samples=args.metrics_eval_samples,
        max_new_tokens=args.metrics_max_new_tokens,
        log_artifacts=args.mlflow,
        batch_size=args.metrics_batch_size,
        example_ids=probe_ids,
    )

    def grpo_reward(completions, **kwargs):
        return reward_func(completions, reward_mode=args.reward_mode, **kwargs)

    print(f"GRPO reward={args.reward_mode}, generations={args.num_generations}, "
          f"generation_batch_size={grpo_config.generation_batch_size}, "
          f"unique_prompts_per_generation_batch={grpo_config.generation_batch_size // args.num_generations}",
          flush=True)
    trainer = GRPOTrainer(
        model=args.base_model,
        reward_funcs=grpo_reward,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        # StallStopCallback must come AFTER fc_callback: it reads the generation-based
        # metrics fc_callback adds to the metrics dict in place, and callbacks run in
        # list order.
        callbacks=[fc_callback] + ([StallStopCallback(
            args.stall_metrics, args.stall_check_step, args.stall_min_delta,
        )] if args.stall_check_step > 0 else []),
    )
    fc_callback.trainer = trainer  # see FunctionCallEvalCallback's docstring (train_sft.py) for why

    mlflow_run = contextlib.nullcontext()
    if args.mlflow:
        import mlflow
        from transformers.integrations import MLflowCallback

        if args.mlflow_tracking_uri:
            os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
        if args.mlflow_experiment_name:
            os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment_name

        # `with mlflow_run:` below (not a bare start_run() call) because MLflowCallback
        # only auto-ends a run it started itself -- since we start it here instead (so
        # LoRA hyperparameters can be logged, matching train_sft.py's same reasoning),
        # its on_train_end() never ends it, and it would sit "RUNNING" in the MLflow UI
        # forever after the process exits. See train_sft.py's own version of this same
        # fix for the fuller explanation, verified against transformers'/mlflow's source.
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        if args.mlflow_experiment_name:
            mlflow.set_experiment(args.mlflow_experiment_name)
        # log_system_metrics=True: samples GPU/CPU/RAM usage every 10s (mlflow default)
        # for the run's duration -- see train_sft.py's own version of this same change
        # for the fuller explanation (needs `nvidia-ml-py` installed for GPU metrics
        # specifically). Especially relevant here: rollout generation is the memory
        # spike train_sft.py's system metrics can't show, since GRPO's is a separate
        # failure mode (already hit OOM + eviction on this job once -- see
        # grpo-smoke-job.yaml's memory limits).
        mlflow_run = mlflow.start_run(log_system_metrics=True)
        mlflow.log_params(
            {
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "kl_beta": args.kl_beta,
                "num_generations": args.num_generations,
                "reward_mode": args.reward_mode,
                "pilot": args.pilot,
                "train_targets": len(train_dataset),
                "validation_targets": len(eval_dataset),
            }
        )

        trainer.add_callback(MLflowCallback())

    with mlflow_run:
        trainer.train()
        # Merge the LoRA adapter into the base weights and save a plain, full model --
        # same reasoning as train_sft.py's identical step: --base-model here is already
        # a merged SFT checkpoint (see that script), and GRPOTrainer's own peft_config
        # wraps it in a *fresh* adapter for this stage, so trainer.model is a PeftModel
        # again by the time training finishes. Leaving it unmerged would just move the
        # same "adapter-only checkpoint, unusable as a standalone model" problem one
        # stage downstream instead of fixing it. trainer.model here is the *best*
        # checkpoint by fc_call_correctness (load_best_model_at_end above), not
        # necessarily the final step's -- same as train_sft.py.
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(args.output_dir))
        # train_grpo.py has no separate tokenizer object (GRPOTrainer builds its own
        # internally from --base-model) -- reload it here purely to save alongside the
        # merged model, same as train_sft.py does with the one it already has in hand.
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(args.base_model).save_pretrained(str(args.output_dir))
        # Trainer's own periodic checkpoints (output_dir/checkpoint-<step>/, adapter +
        # optimizer/scheduler/rng state for resuming) are redundant now that the final
        # state is already merged and saved above -- see train_sft.py's identical
        # cleanup step for why leaving them in place is actively confusing, not just
        # extra disk usage.
        for checkpoint_dir in Path(args.output_dir).glob("checkpoint-*"):
            if checkpoint_dir.is_dir():
                shutil.rmtree(checkpoint_dir)
        if args.mlflow:
            # trainer.save_model() only writes to local disk -- MLflowCallback's own
            # artifact upload is gated behind the HF_MLFLOW_LOG_ARTIFACTS env var
            # (unset here) and only fires on the Trainer's own periodic checkpoint
            # saves, never on this final save call. Without this explicit upload,
            # nothing reaches MLflow's Artifacts tab.
            mlflow.log_artifacts(str(args.output_dir), artifact_path="model")


if __name__ == "__main__":
    main()
