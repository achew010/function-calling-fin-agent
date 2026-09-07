# 1. SFT

**Why SFT first, not GRPO first:** SFT is what teaches the base model the schema and
format at all — RL (see `../2_grpo/`) needs a policy that already produces a reasonable
fraction of valid completions to get useful reward signal from. This is also the simpler
of the two loops (one forward/backward pass per batch, no rollout generation), so it's
the right place to establish the pipeline before adding RL's extra moving parts.

## Base model

**Qwen/Qwen3-8B** — a newer-generation model than the 7B-class ToolACE's own reference
fine-tunes use, and explicitly trained with stronger agentic/tool-use capability as a
design goal; still the same size class the H100/32-concurrency target in `4_deployment/`
is sized against. Swappable via `--base-model`, but **must match the model
`2_evaluations/run_bfcl_eval.py` baselines** — a baseline-vs-fine-tuned comparison is
only meaningful if both measurements start from the same base checkpoint, not just the
same size class, so change both defaults together.

Qwen3 is a hybrid reasoning model (it can emit `<think>...</think>` before answering).
See the note in `train_sft.py`'s docstring for how that interacts with training data,
and `2_evaluations/README.md` for why `enable_thinking` has to be explicitly disabled at
inference/eval time — left on, it inflates latency with chain-of-thought the production
SLA has no room for.

## Method

LoRA via `peft` + `trl.SFTTrainer`, reading the `train`/`val` splits produced by
`0_data/prepare_dataset.py`. QLoRA (4-bit base model) is available via `--use-qlora` if
memory during training is tighter than the target GPU allows.

Each ToolACE conversation is converted into the base model's chat template: `system`
turns carry the tool definitions, `user`/`assistant` turns pass through, and `tool` turns
(function results) are mapped to the template's tool-response role. Assistant turns that
are function calls (parsed the same way `0_data/prepare_dataset.py` does) are re-rendered
as the target JSON schema the production graph's validator expects, not left in ToolACE's
native `Name(arg=val)` call syntax — the model must learn to emit the schema the internal
APIs actually consume.

**Per-workflow adapters:** ToolACE has no fraud/transaction workflow labels, so this
reference run trains a single general adapter on the full prepared dataset. Once the
`add_internal_examples()` hook in `0_data/` carries real internal-API examples tagged by
workflow, `--data-filter` can restrict a training run to one workflow's examples to
produce a second adapter without duplicating the base model.

## Validation metrics

Beyond `eval_loss`, `train_sft.py` reports generation-based function-calling metrics
during validation (`metrics.py`, wired in via `FunctionCallEvalCallback`) — chosen after
reviewing a broader wishlist of possible metrics against what actually applies to a
free-generation (not closed-set classification) model:

- **Tool selection** — exact match on the predicted tool *set* (handles single and
  parallel calls the same way, via set comparison).
- **Parameter extraction** — per-parameter value accuracy, per-parameter type accuracy
  (matters for API compatibility even when a value "looks" right — `"5"` vs `5`), and a
  token-overlap F1 as a smoother, non-binary signal for early training when exact-match
  rates are mostly zero.
- **Composed** — full-call accuracy (tool *and* params both correct) and its
  conditional decomposition, `P(params correct | tool correct)`, to see whether errors
  concentrate in tool selection or in how the tool gets called.
- **Multi-step** — step-level accuracy (free — one instance already *is* one turn) and
  full-trajectory accuracy for multi-turn conversations (every turn in the conversation
  correct, not just each turn scored independently).
- **Error-type breakdown** — `wrong_tool`, `hallucinated_tool`, `missed_call`,
  `unwarranted_call`, `missing_param`, `extra_param`, `wrong_param_value`,
  `wrong_param_type` — a confusion-matrix-style rate per category instead of one flat
  accuracy number, so a training run shows *where* the model is weak, not just *how*
  weak.

Deliberately **not** included, with reasons (see `metrics.py`'s module docstring for the
full rationale): classification-style "Top-N accuracy" doesn't map cleanly onto a model
that free-generates a call rather than scoring over a fixed candidate set — the honest
analogue is pass@N, which costs N× the generation and belongs in a dedicated eval run,
not this per-epoch callback; and "wrong call order" isn't modeled, since matching is
call-name-keyed and order-independent everywhere else in this project.

**Why a callback, not `compute_metrics`:** `Trainer`'s default eval loop is
teacher-forced (predicts the next token given the *correct* prefix so far), which
systematically looks more accurate than real generation — it can't surface the
compounding errors a model makes decoding on its own. `FunctionCallEvalCallback` runs
real `model.generate()` (greedy) on a small, fixed subsample of the validation set
(`--metrics-eval-samples`, default 50) so these numbers mean what
`2_evaluations/run_internal_eval.py`'s numbers mean — deliberately a subsample, not the
full val set, since generation is far more expensive than the forward pass Trainer's own
loss-eval uses. Logs the full `metrics.py` breakdown each eval round (tool-selection
accuracy, param value/type accuracy, hallucination rate, per-error-type rates), not just
the single `eval_fc_call_correctness` scalar `metric_for_best_model` uses — useful for
telling *which* kind of error is driving a change in that scalar rather than just that
one happened.

Verified without a GPU the same way the rest of this project has been: fed a "perfect"
prediction (the exact ground-truth call, or a plausible refusal) through `metrics.py`
for every instance in all three splits — every split scores 1.0 across every accuracy
metric with zero errors. That check itself caught a real bug: the first pass paired
predicted-to-expected calls by function name, which silently mis-paired parallel calls
to the *same* tool with different arguments (e.g. generating several payment cards in
one turn) — now detected and excluded from the per-parameter breakdown rather than
guessed at (see the comment in `metrics.py`'s `score_detailed`).

## MLflow logging

`--mlflow` adds transformers' built-in `MLflowCallback` to the trainer (`--mlflow-experiment-name` / `--mlflow-tracking-uri` set the usual MLflow env vars beforehand; tracking defaults to a local `./mlruns` if unset). This picks up both the Trainer's own metrics (loss, lr, etc.) and `FunctionCallEvalCallback`'s metrics — the latter calls `trainer.log(...)` directly rather than returning values through `compute_metrics`, because `Trainer.evaluate()` calls `self.log(output.metrics)` *before* dispatching to callbacks (verified against the installed transformers source), so mutating the metrics dict inside a callback would silently never reach MLflow or any other logger.

## Run

```bash
python train_sft.py \
  --base-model Qwen/Qwen3-8B \
  --train-file ../../0_data/data/train.jsonl \
  --val-file ../../0_data/data/val.jsonl \
  --output-dir checkpoints/adapter-general \
  --epochs 1 --lr 2e-4 --lora-r 16 --lora-alpha 32 \
  --metrics-eval-samples 50 \
  --mlflow --mlflow-experiment-name fin-agent-sft
```

This checkpoint is the starting policy for `../2_grpo/`.
