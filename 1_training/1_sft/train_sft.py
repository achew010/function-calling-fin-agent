"""LoRA/QLoRA fine-tune of a function-calling base model on the prepared ToolACE splits.

Default base model: Qwen/Qwen3-8B — matches 2_evaluations/run_bfcl_eval.py's default
baseline model. Keeping these in sync matters: a baseline-vs-fine-tuned comparison is
only meaningful if both measurements are the same base checkpoint, not just the same
size class.

Note on Qwen3's "thinking" mode: it's a hybrid reasoning model that can emit
<think>...</think> before its answer. Training data here has no reasoning_content, so
the chat template renders assistant turns with an *empty* think block
(<think>\\n\\n</think>\\n\\n) rather than none at all — this is expected and desired, not
a bug: it's what teaches the model to reproduce the same immediate-answer, non-thinking
pattern used at inference (see the enable_thinking=False notes in 2_evaluations/).

Usage:
    python train_sft.py \
        --base-model Qwen/Qwen3-8B \
        --train-file ../../0_data/data/train.jsonl \
        --val-file ../../0_data/data/val.jsonl \
        --output-dir checkpoints/adapter-general
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

# Reuse the exact call-syntax parser 0_data/prepare_dataset.py uses, rather than
# re-implementing ToolACE's `Name(arg=val)` parsing a second time.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "0_data"))
from prepare_dataset import try_parse_calls  # noqa: E402

# Reuse the eval's instance-construction and scoring logic — the training-time metrics
# below and the eval-gate numbers should be computed by one piece of logic, not two.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "2_evaluations"))
from run_internal_eval import build_eval_instances, normalize_calls, parse_prediction  # noqa: E402

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

from metrics import call_correctness, score_detailed

CALL_TYPES = ["single", "parallel", "multi_turn", "no_call"]


def render_assistant_turn(content: str) -> str:
    """Re-render a ToolACE `Name(arg=val)` call turn as the target JSON call schema.

    The production agentic graph's schema validator (see the top-level README) expects
    structured JSON, not ToolACE's native call syntax — the model has to learn to emit
    what production actually consumes, not the dataset's own notation.
    """
    calls = try_parse_calls(content)
    if calls is None:
        return content
    return json.dumps([{"name": c["name"], "arguments": c["arguments"]} for c in calls])


def load_examples(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def to_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": example["system"]}]
    for turn in example["turns"]:
        content = turn["content"]
        if turn["role"] == "assistant":
            content = render_assistant_turn(content)
        messages.append({"role": turn["role"], "content": content})
    return messages


def build_dataset(path: Path, tokenizer, data_filter: str | None) -> Dataset:
    examples = load_examples(path)
    if data_filter:
        examples = [ex for ex in examples if ex.get("call_type") == data_filter]
    texts = [
        tokenizer.apply_chat_template(
            to_messages(ex), tokenize=False, add_generation_prompt=False
        )
        for ex in examples
    ]
    return Dataset.from_dict({"text": texts})


class FunctionCallEvalCallback(TrainerCallback):
    """Generation-based validation metrics (see metrics.py), reported alongside the
    Trainer's own teacher-forced eval loss.

    Why this can't just be a `compute_metrics` function: transformers.Trainer's default
    eval loop is teacher-forced (next-token predictions conditioned on the
    ground-truth prefix), which systematically looks more accurate than real
    generation — it can't surface the compounding errors a model makes decoding on its
    own. To report numbers that mean what 2_evaluations/run_internal_eval.py's numbers
    mean, this callback runs real `model.generate()` (greedy) on a small, fixed
    subsample of the validation set — intentionally not the full eval set, and
    intentionally not on every batch, since generation is far more expensive than the
    forward pass Trainer's own eval loop uses. Keep `eval_samples` small (tens, not
    hundreds).

    Needs `self.trainer` set after Trainer construction (see main()): Trainer.evaluate()
    calls `self.log(output.metrics)` *before* dispatching `on_evaluate` to callbacks
    (verified against the installed transformers version's source — not assumed), so
    mutating the `metrics` dict here would silently never reach any *logger* (MLflow
    included) for this eval round — that already-fired self.log() call is what a logger
    actually reacts to. Calling `self.trainer.log(...)` directly triggers a fresh,
    correctly dispatched log event instead, which is why that's still done below.

    Separately, and non-obviously, mutating `metrics` here IS still necessary for
    `load_best_model_at_end`/`metric_for_best_model` to be able to use this value at
    all: `_maybe_log_save_evaluate()` calls `self._determine_best_metric(metrics=...)`
    using the exact dict object `evaluate()` returns, and that dict is the same object
    passed into `on_evaluate` here (verified against source: `on_evaluate` runs, then
    `evaluate()` returns that same `output.metrics` reference, before the outer loop
    ever inspects it) — an in-place mutation here is visible to that later check, even
    though a plain `self.trainer.log(...)` call is a separate, disconnected event.
    """

    def __init__(
        self, val_file: Path, eval_samples: int = 30, max_new_tokens: int = 256, seed: int = 0
    ) -> None:
        examples = load_examples(val_file)
        rng = random.Random(seed)
        self.examples = rng.sample(examples, min(eval_samples, len(examples)))
        self.max_new_tokens = max_new_tokens
        self.trainer: Any = None  # set by main() after Trainer construction

    def on_evaluate(self, args, state, control, metrics, model=None, processing_class=None, **kwargs):
        if model is None or processing_class is None or self.trainer is None:
            return
        tokenizer = processing_class
        was_training = model.training
        model.eval()
        scored_examples: list[tuple[str, list[dict[str, Any]]]] = []
        with torch.no_grad():
            for ex in self.examples:
                results = []
                for instance in build_eval_instances(ex):
                    prompt = tokenizer.apply_chat_template(
                        instance["context"],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    completion_ids = output_ids[0][inputs["input_ids"].shape[1] :]
                    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                    results.append(score_detailed(instance, completion_text, parse_prediction, normalize_calls))
                scored_examples.append((ex["call_type"], results))
        if was_training:
            model.train()

        fc_correctness = call_correctness(scored_examples)
        # Default to 0.0 (rather than leaving the key absent) when this round's sample
        # happened to contain zero call-cases: metric_for_best_model="fc_call_correctness"
        # (see main()) needs this key present on every eval round, or
        # _determine_best_metric raises KeyError the first time it's missing.
        metrics["eval_fc_call_correctness"] = fc_correctness if fc_correctness is not None else 0.0
        self.trainer.log({"eval_fc_call_correctness": metrics["eval_fc_call_correctness"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data-filter",
        default=None,
        choices=CALL_TYPES,
        help=(
            "Restrict training to one call_type. Placeholder for training a "
            "workflow-specific adapter (fraud vs. transaction) once 0_data's "
            "add_internal_examples() hook carries real workflow-tagged examples."
        ),
    )
    parser.add_argument("--use-qlora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help=(
            "Attention-only (q/k/v/o_proj) is a common but underpowered default for a "
            "task like this: teaching the model a new *output format* (JSON call "
            "schema instead of ToolACE's native syntax) leans heavily on the MLP "
            "blocks, so gate/up/down_proj are included by default too."
        ),
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Overrides --epochs when > 0 — e.g. for a smoke run of a fixed number of steps regardless of dataset size.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Fraction of total steps to linearly warm up the LR over, before the "
        "scheduler's normal decay — without it, training starts at the full "
        "--lr (2e-4 by default) from step 1, a known source of early instability.",
    )
    parser.add_argument(
        "--per-device-batch-size",
        type=int,
        default=16,
        help="16 (not 2) by default: LoRA's memory footprint (frozen backbone + a "
        "small adapter) leaves an H100 with plenty of headroom, so a larger "
        "per-device batch cuts wall-clock time. Paired with --grad-accum 1 below "
        "for an effective batch size of 16 with no accumulation.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="1 (not 8) by default — --per-device-batch-size above already reaches "
        "the target effective batch size on its own; raise this instead of "
        "--per-device-batch-size if 16 turns out to be too large for the GPU's memory.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="8192 (not 4096) for headroom on longer conversations/tool schemas — "
        "well within Qwen3-8B's native 40960 context. Verified against the real "
        "training data (9515 examples): p99 ~2160 tokens, max ~4479 — 4096 was not "
        "actually truncating meaningfully, so 8192 is comfortable margin over the "
        "observed max, not a fix for an observed problem.",
    )
    parser.add_argument(
        "--eval-strategy",
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="'epoch' can silently never fire the eval loop when --max-steps stops training before one epoch completes — use 'steps' + --eval-steps for a smoke run.",
    )
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--metrics-eval-samples",
        type=int,
        default=30,
        help="Validation examples to run real generation-based metrics on per eval (see FunctionCallEvalCallback) — kept small since generation is expensive.",
    )
    parser.add_argument("--metrics-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--mlflow", action="store_true", help="Log metrics to MLflow via transformers' built-in MLflowCallback."
    )
    parser.add_argument("--mlflow-experiment-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None, help="Defaults to local ./mlruns if unset.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if args.use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype="bfloat16",
    )
    if args.use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = build_dataset(args.train_file, tokenizer, args.data_filter)
    val_dataset = build_dataset(args.val_file, tokenizer, args.data_filter)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_seq_length,  # trl renamed SFTConfig's max_seq_length -> max_length in newer releases; CLI flag name kept for stability
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        # Makes the final trainer.save_model() (and the mlflow.log_artifacts() upload
        # after it, below) save/upload the best-scoring checkpoint by real generation-
        # based call correctness, not just whatever the last step happened to produce —
        # Trainer reloads state.best_model_checkpoint into self.model at the end of
        # train() when this is set. Requires eval_strategy == save_strategy (and, for
        # "steps", save_steps a multiple of eval_steps) — already true of both this
        # script's own defaults (epoch/epoch) and the smoke test's overrides
        # (steps/steps, 10 % 5 == 0); see TrainingArguments' own validation for why.
        load_best_model_at_end=True,
        metric_for_best_model="fc_call_correctness",
        greater_is_better=True,
        bf16=True,
        dataset_text_field="text",
        report_to=[],
    )

    fc_callback = FunctionCallEvalCallback(
        args.val_file,
        eval_samples=args.metrics_eval_samples,
        max_new_tokens=args.metrics_max_new_tokens,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[fc_callback],
    )
    fc_callback.trainer = trainer  # see FunctionCallEvalCallback docstring for why

    mlflow_run = contextlib.nullcontext()
    if args.mlflow:
        import mlflow
        from transformers.integrations import MLflowCallback

        if args.mlflow_tracking_uri:
            os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri
        if args.mlflow_experiment_name:
            os.environ["MLFLOW_EXPERIMENT_NAME"] = args.mlflow_experiment_name

        # MLflowCallback (added below) auto-logs TrainingArguments + model.config, but
        # model.config on a PeftModel is a passthrough to the *frozen base model's*
        # config (verified against peft's source) -- it never sees the LoraConfig, which
        # PEFT keeps on a separate `peft_config` attribute. MLflowCallback.setup() only
        # calls mlflow.start_run() when no run is already active (verified against
        # transformers' source), so starting the run here first and logging the LoRA
        # hyperparameters onto it is the only way they end up in MLflow at all -- without
        # this, a run would show every TrainingArguments field but not r/alpha/dropout/
        # target_modules, the exact values that actually define what got fine-tuned.
        #
        # Also why this is `with mlflow_run:` below rather than a bare start_run() call:
        # MLflowCallback only auto-ends a run it started itself (_auto_end_run, set in
        # its setup() only on the branch that calls start_run() -- verified against its
        # source). Since this run is started here instead, on_train_end() never ends
        # it, and it would sit "RUNNING" in the MLflow UI forever after the process
        # exits. mlflow.start_run()'s return value is itself a context manager that
        # ends the run (FINISHED, or FAILED on an exception) on __exit__.
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        if args.mlflow_experiment_name:
            mlflow.set_experiment(args.mlflow_experiment_name)
        mlflow_run = mlflow.start_run()
        mlflow.log_params(
            {
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target_modules": ",".join(args.target_modules),
                "use_qlora": args.use_qlora,
            }
        )

        trainer.add_callback(MLflowCallback())

    with mlflow_run:
        trainer.train()
        trainer.save_model(str(args.output_dir))
        tokenizer.save_pretrained(str(args.output_dir))
        if args.mlflow:
            # The two save calls above only write to local disk. MLflowCallback does
            # have its own artifact upload (on_save), but it's gated behind the
            # HF_MLFLOW_LOG_ARTIFACTS env var (nothing here sets it) and only fires on
            # the Trainer's own periodic checkpoint saves during training -- never on
            # this final save_model() call, which isn't part of the Trainer's
            # instrumented save path. Without this explicit upload, nothing ever
            # reaches MLflow's Artifacts tab regardless of --save-strategy/--save-steps.
            mlflow.log_artifacts(str(args.output_dir), artifact_path="model")


if __name__ == "__main__":
    main()
