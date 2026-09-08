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
        "sets steps_per_generation (TRL defaults it to gradient_accumulation_steps), "
        "i.e. how many optimizer steps reuse one batch of rollouts.",
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
        default=50,
        help="Validation examples FunctionCallEvalCallback runs real generation-based "
        "metrics on per eval -- see train_sft.py's identical flag.",
    )
    parser.add_argument("--metrics-max-new-tokens", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--mlflow", action="store_true", help="Log metrics to MLflow via transformers' built-in MLflowCallback."
    )
    parser.add_argument("--mlflow-experiment-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None, help="Defaults to local ./mlruns if unset.")
    args = parser.parse_args()

    train_dataset = build_grpo_dataset(args.train_file)
    # Also gives GRPOTrainer's own built-in eval loop (reward/kl/entropy/completion-
    # length, computed the same way as training rollouts) on real held-out data for
    # free, on top of FunctionCallEvalCallback's richer metrics.py breakdown below --
    # both need a non-None eval_dataset to fire at all (Trainer raises otherwise).
    eval_dataset = build_grpo_dataset(args.val_file)

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
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        # Mirrors train_sft.py: reload the checkpoint with the best real,
        # generation-based fc_call_correctness (FunctionCallEvalCallback below) at the
        # end of train(), rather than merging/saving whatever the final step produced --
        # this is the actual fix for GRPO previously having no protection against a
        # reward-optimized policy drifting away from real correctness over the run.
        load_best_model_at_end=True,
        metric_for_best_model="fc_call_correctness",
        greater_is_better=True,
        report_to=[],
    )

    fc_callback = FunctionCallEvalCallback(
        args.val_file,
        eval_samples=args.metrics_eval_samples,
        max_new_tokens=args.metrics_max_new_tokens,
    )

    trainer = GRPOTrainer(
        model=args.base_model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        callbacks=[fc_callback],
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
