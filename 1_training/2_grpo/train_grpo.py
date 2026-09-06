"""GRPO fine-tune of a function-calling policy, via TRL's GRPOTrainer.

Starting point: the SFT checkpoint from ../1_sft/ (see that README for why RL needs a
warm-started policy rather than the raw base model — GRPO gets little useful reward
signal from a policy that rarely produces a valid completion to begin with).

Reward: mirrors 2_evaluations/run_internal_eval.py's scoring dimensions (function-name
match, parameter match, hallucination penalty, refusal correctness), collapsed into one
scalar with partial credit — see compute_reward. This is what "training with BFCL/the
internal eval in mind" means concretely: the reward function and the eval gate are the
same logic, so training optimizes directly toward what promotion is gated on, rather
than token-level match against ToolACE's one reference completion. It's a starting
design, not a final one — see the README for the risk-tier-weighting extension once
0_data's risk_tier field carries real values instead of "unclassified".

Dataset: reuses 2_evaluations/run_internal_eval.py's build_eval_instances (imported, not
reimplemented) to turn each conversation into one training prompt per assistant turn —
the exact same instances the internal eval scores checkpoints against.

Usage:
    python train_grpo.py \
        --base-model ../1_sft/checkpoints/adapter-general \
        --train-file ../../0_data/data/train.jsonl \
        --output-dir checkpoints/grpo-general
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Reuse the exact instance-construction and scoring logic 2_evaluations/run_internal_eval.py
# uses, rather than re-implementing it — the reward function and the eval gate should stay
# one piece of logic, not two that can quietly drift apart.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "2_evaluations"))
from run_internal_eval import build_eval_instances, load_examples, normalize_calls, parse_prediction  # noqa: E402

from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer


def build_grpo_dataset(path: Path) -> Dataset:
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


def compute_reward(
    text: str, tool_names: list[str], is_call_case: bool, expected_calls: list[dict[str, Any]]
) -> float:
    predicted = parse_prediction(text)
    if not is_call_case:
        return 1.0 if predicted is None else -1.0  # correct refusal vs. an unwarranted call
    if predicted is None:
        return -1.0  # missed a call it should have made
    if any(c["name"] not in tool_names for c in predicted):
        return -2.0  # hallucinated a call — worst case, matches the risk-tiering framing
    if {c["name"] for c in predicted} != {c["name"] for c in expected_calls}:
        return -0.5  # wrong function entirely
    return 1.0 if normalize_calls(predicted) == normalize_calls(expected_calls) else 0.3  # right fn, wrong args


def reward_func(
    completions: list[list[dict[str, str]]],
    tool_names: list[list[str]],
    is_call_case: list[bool],
    expected_calls: list[str],
    **kwargs: Any,
) -> list[float]:
    """GRPOTrainer's reward-function contract: `completions` is one message per
    generation (`[{"content": "..."}]`); every other training-dataset column is passed
    through as a same-length list, aligned to `completions` (verified against the
    installed trl version's GRPOTrainer/trl.rewards source — not assumed). `expected_calls`
    arrives JSON-encoded (see build_grpo_dataset) and is decoded here."""
    texts = [c[0]["content"] for c in completions]
    return [
        compute_reward(text, names, is_case, json.loads(calls_json))
        for text, names, is_case, calls_json in zip(texts, tool_names, is_call_case, expected_calls)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-model",
        default="../1_sft/checkpoints/adapter-general",
        help="Starting policy — the SFT checkpoint from ../1_sft/, not the raw base model (see module docstring).",
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-generations", type=int, default=8, help="Group size G — completions sampled per prompt.")
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--kl-beta",
        type=float,
        default=0.0,
        help="KL penalty against the reference policy. TRL's current default (0.0, DAPO-style) "
        "relies on loss clipping alone; set e.g. 0.04 for classic GRPO-style KL regularization "
        "if the policy drifts too far from the SFT checkpoint.",
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Overrides --epochs when > 0 — e.g. for a smoke run of a fixed number of steps regardless of dataset size.",
    )
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--mlflow", action="store_true", help="Log metrics to MLflow via transformers' built-in MLflowCallback."
    )
    parser.add_argument("--mlflow-experiment-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None, help="Defaults to local ./mlruns if unset.")
    args = parser.parse_args()

    train_dataset = build_grpo_dataset(args.train_file)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    grpo_config = GRPOConfig(
        output_dir=str(args.output_dir),
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.kl_beta,
        # Qwen3 is a hybrid reasoning model — same non-thinking requirement as everywhere
        # else in this project (2_evaluations/README.md), so rollouts during training
        # match the behavior the model is actually served with.
        chat_template_kwargs={"enable_thinking": False},
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=args.base_model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )

    if args.mlflow:
        import os

        from transformers.integrations import MLflowCallback

        if args.mlflow_tracking_uri:
            os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
        if args.mlflow_experiment_name:
            os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment_name
        trainer.add_callback(MLflowCallback())

    trainer.train()
    trainer.save_model(str(args.output_dir))


if __name__ == "__main__":
    main()
