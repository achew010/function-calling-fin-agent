"""LoRA/QLoRA fine-tune of a function-calling base model on the prepared ToolACE splits.

Default base model: Qwen/Qwen3-8B — matches 2_evaluations/run_bfcl_eval.py's default
baseline model. Keeping these in sync matters: a baseline-vs-fine-tuned comparison is
only meaningful if both measurements are the same base checkpoint, not just the same
size class.

Each assistant turn is a separate prompt/completion example. The prompt contains all
preceding conversation history; only the target assistant response and its end-of-turn
marker contribute to the loss. Qwen3's empty think block is supplied in the prompt
with enable_thinking=False, matching generation-based evaluation and inference.

Evaluation runs at step zero before the first optimizer update, then at the configured
cadence. Both the teacher-forced loss and generation-based metrics include this baseline.

Saves a merged, standalone model to --output-dir (LoRA folded into the base weights via
merge_and_unload(), not an adapter-only checkpoint) — see main()'s comment at the save
step for why: every downstream consumer (train_grpo.py, run_bfcl_eval.py) loads
checkpoints the same way it loads the raw Qwen/Qwen3-8B baseline, no adapter-aware
loading path needed anywhere else.

Usage:
    python train_sft.py \
        --base-model Qwen/Qwen3-8B \
        --train-file ../../0_data/data/train.jsonl \
        --val-file ../../0_data/data/val.jsonl \
        --output-dir checkpoints/sft-general
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Reuse the eval's instance-construction and scoring logic — the training-time metrics
# below and the eval-gate numbers should be computed by one piece of logic, not two.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "2_evaluations"))
from run_internal_eval import build_eval_instances, normalize_calls, parse_prediction  # noqa: E402

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

from metrics import aggregate_metrics, compare_validation_reports, score_detailed, summarize_error_gaps

CALL_TYPES = ["single", "parallel", "multi_turn", "no_call"]


# NOTE: assistant turns train on ToolACE's native `[Name(arg=val)]` syntax, deliberately
# unmodified -- system messages carry the tool list and its instruction to answer in
# that exact format, so training on it too is what makes the two agree (a prior version
# re-rendered turns as JSON here, which produced a fine-tune that disobeyed its own
# system prompt on every single example -- see 0_data/README.md's verification section).
# 0_data/prepare_dataset.py now also makes that native syntax genuinely BFCL-parseable
# (real Python identifiers, not ToolACE's free-text tool/parameter names), via
# rename_tools_for_bfcl -- so this file no longer needs try_parse_calls itself; the
# renaming already happened once, at dataset-prep time, not per training run.
#
# Production still needs JSON: that conversion belongs downstream of the model
# (try_parse_calls + json.dumps, the same two lines as before) rather than baked into
# the training target.


def load_examples(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def to_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": example["system"]}]
    for turn in example["turns"]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    return messages


def build_dataset(path: Path, tokenizer, data_filter: str | None) -> Dataset:
    examples = load_examples(path)
    if data_filter:
        examples = [ex for ex in examples if ex.get("call_type") == data_filter]
    # One target assistant turn per row. Earlier assistant/tool turns remain in the
    # prompt as context, but only this turn contributes to the loss. Render here so
    # Qwen's non-thinking generation prefix is identical to the evaluation prefix;
    # this also avoids depending on a chat template with assistant-mask support.
    prompts, completions = [], []
    for example_index, ex in enumerate(examples):
        messages = to_messages(ex)
        for turn_index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prompt = tokenizer.apply_chat_template(
                messages[:turn_index],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full_text = tokenizer.apply_chat_template(
                messages[: turn_index + 1],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            # Fail rather than silently supervise part of the prompt if a different
            # model's chat template renders history differently with a target added.
            if not full_text.startswith(prompt) or len(full_text) == len(prompt):
                raise ValueError(
                    f"{path}: example {example_index}, turn {turn_index}: "
                    "chat template must render a nonempty completion after the "
                    "non-thinking generation prefix"
                )
            prompts.append(prompt)
            completion = full_text[len(prompt) :]
            # Qwen's template leaves a newline after <|im_end|>. TRL appends EOS
            # unless the string ends with eos_token, which otherwise creates a
            # second end marker. Strip only whitespace AFTER an existing EOS.
            if tokenizer.eos_token and completion.rstrip().endswith(tokenizer.eos_token):
                completion = completion.rstrip()
            completions.append(completion)
    if not prompts:
        raise ValueError(f"{path}: no assistant targets after filtering")
    return Dataset.from_dict({"prompt": prompts, "completion": completions})


class FunctionCallEvalCallback(TrainerCallback):
    """Generation-based validation metrics (see metrics.py), reported alongside the
    Trainer's own teacher-forced eval loss.

    Why this can't just be a `compute_metrics` function: transformers.Trainer's default
    eval loop is teacher-forced (next-token predictions conditioned on the
    ground-truth prefix), which systematically looks more accurate than real
    generation — it can't surface the compounding errors a model makes decoding on its
    own. To report numbers that mean what 2_evaluations/run_internal_eval.py's numbers
    mean, this callback runs real `model.generate()` (greedy) on the full validation
    set by default. Positive eval_samples opts into a fixed subset for smoke tests.
    Per-turn predictions are saved at every evaluation; subsequent evaluations are
    compared with step zero using a paired bootstrap clustered by conversation.

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
        self, val_file: Path, eval_samples: int = 0, max_new_tokens: int = 256, seed: int = 0,
        log_artifacts: bool = False, batch_size: int = 4, example_ids: list[int] | None = None,
    ) -> None:
        if eval_samples < 0:
            raise ValueError("eval_samples must be nonnegative (0 means full validation)")
        if batch_size < 1:
            raise ValueError("Generation batch_size must be positive")
        self.batch_size = batch_size
        examples = load_examples(val_file)
        if not examples:
            raise ValueError("Validation dataset is empty")
        self.dataset_sha256 = hashlib.sha256(val_file.read_bytes()).hexdigest()
        self.scorer_sha256 = hashlib.sha256(b"\n".join(
            Path(sys.modules[name].__file__).read_bytes()
            for name in ("metrics", "run_internal_eval", "prepare_dataset")
        )).hexdigest()
        rng = random.Random(seed)
        indexed = list(enumerate(examples))
        self.examples = rng.sample(indexed, min(eval_samples, len(indexed))) if eval_samples else indexed
        if example_ids is not None:
            if not example_ids or len(set(example_ids)) != len(example_ids) or any(i < 0 or i >= len(indexed) for i in example_ids):
                raise ValueError("Validation example_ids must be nonempty, unique, and in range")
            self.examples = [indexed[i] for i in example_ids]
        self.max_new_tokens = max_new_tokens
        self.log_artifacts = log_artifacts
        self.baseline_report = None
        self.trainer: Any = None  # set by main() after Trainer construction

    def on_train_begin(self, args, state, control, **kwargs):
        if args.eval_on_start and args.process_index == 0:
            print("[validation] Starting step-zero loss evaluation; full generation evaluation follows "
                  "before the first training update.", flush=True)

    def on_evaluate(self, args, state, control, metrics, model=None, processing_class=None, **kwargs):
        if model is None or processing_class is None or self.trainer is None:
            return
        tokenizer = processing_class
        was_training = model.training
        padding_side = tokenizer.padding_side
        scored_examples: list[tuple[str, list[dict[str, Any]]]] = []
        conversations = [{"conversation_id": cid, "call_type": ex["call_type"], "turns": []}
                         for cid, ex in self.examples]
        pending = [(i, turn_index, instance)
                   for i, (_, ex) in enumerate(self.examples)
                   for turn_index, instance in enumerate(build_eval_instances(ex))]
        started = time.monotonic()
        eos_ids = model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = tokenizer.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else (eos_ids or [])
        try:
            model.eval()
            tokenizer.padding_side = "left"  # decoder-only batched generation
            with torch.no_grad():
                for offset in range(0, len(pending), self.batch_size):
                    batch = pending[offset: offset + self.batch_size]
                    if args.process_index == 0:
                        print(f"[validation step={state.global_step}] generating turns "
                              f"{offset + 1}-{offset + len(batch)}/{len(pending)} "
                              f"(elapsed {time.monotonic() - started:.1f}s)", flush=True)
                    prompts = [tokenizer.apply_chat_template(
                        instance["context"], tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    ) for _, _, instance in batch]
                    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_ids or None,
                        # gradient_checkpointing=True (see SFTConfig below) makes
                        # transformers force model.config.use_cache=False for training,
                        # and nothing re-enables it for eval -- without this override,
                        # every one of these generate() calls recomputes the full
                        # sequence from scratch at each new token instead of using a KV
                        # cache. use_cache here is a call-level override (generate()'s
                        # own effective GenerationConfig), so it doesn't need undoing
                        # before model.train() resumes.
                        use_cache=True,
                    )
                    for row, (i, turn_index, instance) in zip(output_ids, batch):
                        completion_ids = row[inputs["input_ids"].shape[1]:].tolist()
                        # Finished sequences are padded to the longest completion in
                        # the batch. Count only through their first stopping token.
                        stop = next((j for j, token in enumerate(completion_ids) if token in eos_ids), None)
                        if stop is not None:
                            completion_ids = completion_ids[:stop + 1]
                        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                        score = score_detailed(instance, completion_text, parse_prediction, normalize_calls)
                        conversations[i]["turns"].append({
                            "turn_index": turn_index, "context": instance["context"],
                            "expected_calls": instance["expected_calls"],
                            "tool_names": sorted(instance["tool_names"]),
                            "prediction": completion_text, "score": score,
                            "generated_tokens": len(completion_ids),
                            "reached_token_limit": stop is None and len(completion_ids) >= self.max_new_tokens,
                        })
        finally:
            tokenizer.padding_side = padding_side
            model.train(was_training)
        for conversation in conversations:
            scored_examples.append((conversation["call_type"], [t["score"] for t in conversation["turns"]]))
        if args.process_index == 0:
            print(f"[validation step={state.global_step}] generation complete: {len(pending)} turns "
                  f"in {time.monotonic() - started:.1f}s; scoring and saving report.", flush=True)

        detailed = aggregate_metrics(scored_examples)
        error_gaps = summarize_error_gaps(conversations, parse_prediction)
        for name, values in error_gaps.items():
            detailed[f"gap_{name}_accuracy"] = values["accuracy"]
            detailed[f"gap_{name}_n"] = values["n"]
            detailed[f"gap_{name}_truncated"] = values["truncated"]
            detailed[f"gap_{name}_wrong_call_count"] = values["wrong_call_count"]
        metrics.update({f"eval_{key}": value for key, value in detailed.items()})
        report = {
            "schema_version": 1, "dataset_sha256": self.dataset_sha256,
            "scorer_sha256": self.scorer_sha256,
            "step": state.global_step,
            "generation": {"do_sample": False, "max_new_tokens": self.max_new_tokens,
                           "batch_size": self.batch_size, "eos_token_ids": eos_ids,
                           "enable_thinking": False,
                           "chat_template_sha256": hashlib.sha256(
                               json.dumps(tokenizer.chat_template, sort_keys=True).encode()).hexdigest()},
            "metrics": detailed, "conversations": conversations, "error_gaps": error_gaps,
        }
        if state.global_step == 0:
            self.baseline_report = report
        elif self.baseline_report is not None:
            report["comparison_to_baseline"] = compare_validation_reports(self.baseline_report, report)
        if args.process_index == 0:
            output = Path(args.output_dir) / "validation_predictions"
            output.mkdir(parents=True, exist_ok=True)
            path = output / f"step-{state.global_step:06d}.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2))
            temporary.replace(path)
            if self.log_artifacts:
                import mlflow
                mlflow.log_artifact(str(path), artifact_path="validation_predictions")
        # full_call_accuracy is the same computation call_correctness() used to do
        # separately (see metrics.py) -- reading it off the aggregate dict instead
        # keeps there being exactly one place that logic lives.
        fc_correctness = detailed.get("full_call_accuracy")
        # Default to 0.0 (rather than leaving the key absent) when this round's sample
        # happened to contain zero call-cases: metric_for_best_model="fc_call_correctness"
        # (see main()) needs this key present on every eval round, or
        # _determine_best_metric raises KeyError the first time it's missing.
        metrics["eval_fc_call_correctness"] = fc_correctness if fc_correctness is not None else 0.0
        # Everything else aggregate_metrics computed (tool-selection accuracy,
        # param value/type accuracy, hallucination rate, per-error-type rates, ...)
        # logged too, not just the single scalar above -- so a run that trends down on
        # fc_call_correctness can be diagnosed (which error type is driving it) without
        # having to rerun eval against saved checkpoints after the fact. Excludes
        # full_call_accuracy itself, already covered by eval_fc_call_correctness above.
        self.trainer.log(
            {
                "eval_fc_call_correctness": metrics["eval_fc_call_correctness"],
                **{f"eval_{k}": v for k, v in detailed.items() if k != "full_call_accuracy"},
                **{
                    f"eval_vs_baseline_{name}_{key}": values[key]
                    for name, values in report.get("comparison_to_baseline", {}).get("metrics", {}).items()
                    for key in ("delta", "ci_low", "ci_high")
                },
            }
        )


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
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help=(
            "Attention-only by default (dropped gate/up/down_proj — MLP adapters add "
            "a lot of trainable capacity, and a run showed eval_fc_call_correctness "
            "degrading over training while teacher-forced loss/token-accuracy kept "
            "improving, a pattern consistent with that extra capacity overfitting to "
            "surface token patterns rather than real call correctness). Matches "
            "train_grpo.py's target_modules, which was already attention-only. Add "
            "gate/up/down_proj back if attention-only turns out to be underpowered "
            "for learning the call format."
        ),
    )
    parser.add_argument("--epochs", type=float, default=1.0)
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
        "--lr (2e-4 by default) from step 1, a known source of early instability. "
        "Passed as SFTConfig's warmup_steps below, not warmup_ratio: transformers "
        "removed warmup_ratio entirely as of the 5.x line, consolidating it into "
        "warmup_steps (an int is exact steps, a float in [0, 1) is a ratio of total "
        "steps — verified against the installed transformers version's source, not "
        "assumed). CLI flag name kept as --warmup-ratio for stability.",
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
        "--eval-batch-size",
        type=int,
        default=4,
        help="Smaller than --per-device-batch-size on purpose: Trainer's teacher-forced "
        "eval loop stacks its own forward-pass activations on top of whatever "
        "gradient checkpointing already has resident from training (see sft-job.yaml's "
        "own comment on a real CUDA OOM this caused), and unlike the train batch, "
        "there's no gradient/optimizer state riding on eval throughput to justify a "
        "larger batch here.",
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
        default="steps",
        choices=["no", "steps", "epoch"],
        help="'steps' (not 'epoch') by default so the generation-based eval fires at a "
        "fixed cadence regardless of dataset size or epoch count — 'epoch' can also "
        "silently never fire the eval loop when --max-steps stops training before one "
        "epoch completes. Paired with --eval-steps 140 below.",
    )
    parser.add_argument("--eval-steps", type=int, default=140)
    parser.add_argument(
        "--save-strategy",
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Kept equal to --eval-strategy (both 'steps') — load_best_model_at_end "
        "requires save_strategy == eval_strategy, with save_steps a multiple of "
        "eval_steps (verified against TrainingArguments' own validation).",
    )
    parser.add_argument("--save-steps", type=int, default=140)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--metrics-eval-samples",
        type=int,
        default=0,
        help="Validation conversations for generation metrics: 0 uses the full split (default); a positive number selects a fixed subset for smoke tests.",
    )
    parser.add_argument("--metrics-max-new-tokens", type=int, default=256)
    parser.add_argument("--metrics-batch-size", type=int, default=4,
                        help="Batch size for generation-based validation; reduce to 1 if GPU memory is tight.")
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
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_ratio,  # see --warmup-ratio's help above for why this isn't warmup_ratio
        max_length=args.max_seq_length,  # trl renamed SFTConfig's max_seq_length -> max_length in newer releases; CLI flag name kept for stability
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        # Runs both teacher-forced loss and FunctionCallEvalCallback at step zero,
        # after logger setup but before the first optimizer update.
        eval_on_start=True,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        # Makes the final trainer.save_model() (and the mlflow.log_artifacts() upload
        # after it, below) save/upload the best-scoring checkpoint by real generation-
        # based call correctness, not just whatever the last step happened to produce —
        # Trainer reloads state.best_model_checkpoint into self.model at the end of
        # train() when this is set. Requires eval_strategy == save_strategy (and, for
        # "steps", save_steps a multiple of eval_steps) — already true of both this
        # script's own defaults (steps/steps, 140 % 140 == 0) and the smoke test's
        # overrides (steps/steps, 10 % 5 == 0); see TrainingArguments' own validation for why.
        load_best_model_at_end=True,
        metric_for_best_model="fc_call_correctness",
        greater_is_better=True,
        bf16=True,
        completion_only_loss=True,
        report_to=[],
        # Trades compute for activation memory: without this, training already sits
        # within ~3.6GiB of the 80GiB H100 ceiling at batch 16/seq 8192 (hit a real CUDA
        # OOM on the first epoch-boundary eval, which stacks its own forward-pass
        # activations on top of that). SFTTrainer handles the PEFT-specific wiring
        # (model.enable_input_require_grads(), use_reentrant) internally when this is
        # set on SFTConfig — verified against the installed trl version's source, not
        # assumed.
        gradient_checkpointing=True,
    )

    fc_callback = FunctionCallEvalCallback(
        args.val_file,
        eval_samples=args.metrics_eval_samples,
        max_new_tokens=args.metrics_max_new_tokens,
        log_artifacts=args.mlflow,
        batch_size=args.metrics_batch_size,
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
        # log_system_metrics=True: samples GPU/CPU/RAM usage every 10s (mlflow default)
        # for the run's duration -- needs `nvidia-ml-py` installed (tox.ini) for the
        # gpu_* metrics specifically, or it silently falls back to CPU/RAM/disk only
        # (verified against mlflow's GPUMonitor source). Would have shown VRAM climbing
        # toward the ceiling in real time instead of only finding out via a CUDA OOM
        # traceback after the fact.
        mlflow_run = mlflow.start_run(log_system_metrics=True)
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
        # Merge the LoRA adapter into the base weights and save a plain, full model —
        # not trainer.save_model()'s adapter-only output (adapter_config.json +
        # adapter_model.safetensors, no usable standalone model). Downstream consumers
        # (train_grpo.py's --base-model, run_bfcl_eval.py's --local-model-path) both
        # expect something loadable directly via AutoModelForCausalLM.from_pretrained,
        # not a PEFT adapter needing its own base model resolved separately — merging
        # here means every consumer of this checkpoint uses the exact same loading path
        # as the plain Qwen/Qwen3-8B baseline, adapter-awareness included nowhere else.
        # Not supported for --use-qlora (merging LoRA into a 4-bit-quantized base is a
        # known PEFT limitation — dequantizing first is a separate, unimplemented step);
        # only matters if that flag is ever turned on, off by default.
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(args.output_dir))
        tokenizer.save_pretrained(str(args.output_dir))
        # Trainer's own periodic checkpoints (output_dir/checkpoint-<step>/, containing
        # the pre-merge adapter plus optimizer/scheduler/rng state for resuming
        # training) are redundant now: the best one was already reloaded into
        # trainer.model (load_best_model_at_end, above) and merged into the save just
        # above. Left in place, they'd sit alongside the real deliverable in both
        # output_dir and (via log_artifacts below) MLflow's Artifacts tab, with no clear
        # signal which one is actually "the" model — confirmed this was confusing in
        # practice, not just a hypothetical.
        for checkpoint_dir in Path(args.output_dir).glob("checkpoint-*"):
            if checkpoint_dir.is_dir():
                shutil.rmtree(checkpoint_dir)
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
