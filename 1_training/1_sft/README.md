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
turns carry the tool definitions, and `user`/`assistant`/`tool` turns pass through
unmodified.

Each assistant turn becomes one **prompt/completion** training row. Its prompt contains
the system message and all preceding turns; its completion contains only that assistant
response plus the end-of-turn marker. `completion_only_loss=True` masks the entire
prompt, including earlier assistant turns and tool responses, as well as padding.
No-call responses and clarification questions remain supervised assistant targets.
The prompt uses `enable_thinking=False`, exactly as generation evaluation does; Qwen3's
empty think block belongs to this supplied prefix, not the supervised completion.

This replaces full-conversation language-model loss. With the current local data,
9,515 training conversations become 11,793 assistant targets (validation: 529 → 651).
At batch 16 with accumulation 1 on one GPU, one epoch is now approximately 738 updates
instead of 595. Loss and token accuracy now measure assistant completions, so their
absolute values are not directly comparable to previous full-conversation runs.

**Assistant call turns keep ToolACE's native `[Name(arg=val)]` syntax** — not
re-rendered as JSON. An earlier version did convert them, which was wrong on two counts,
both confirmed against a served checkpoint: it contradicted the data's own system
prompt, which ends every example with `Put it in the format of
[func1(params_name=params_value...)]` / `NO other text MUST be included`, so training
told the model one format and showed it another on every single example; and it
discarded BFCL comparability for nothing, since ToolACE's native syntax is the format
BFCL's own `DEFAULT_SYSTEM_PROMPT` demands (near-identical down to the call template).

That native syntax is now also genuinely BFCL-*parseable*, not just template-compatible
with it: `0_data/prepare_dataset.py`'s `rename_tools_for_bfcl` sanitizes every tool and
parameter name into a real Python identifier (ToolACE's own names routinely aren't —
spaces, apostrophes, hyphens, even bare reserved words like `from`), applied
consistently to both the system prompt's tool list and the calls that reference it. See
`0_data/README.md`'s verification section for the measured before/after (52.0% → 100% of
call turns parseable under BFCL's real `ast_parse`).

Production still needs JSON; that conversion now happens *downstream* of the model
(`try_parse_calls` + `json.dumps`, the same two lines as before) rather than being baked
into the training target. `2_evaluations/run_internal_eval.py`'s `parse_prediction`
reads this same native syntax, so ingestion, training, internal scoring, and BFCL all
agree on one format.

**Per-workflow adapters:** ToolACE has no fraud/transaction workflow labels, so this
reference run trains a single general adapter on the full prepared dataset. Once the
`add_internal_examples()` hook in `0_data/` carries real internal-API examples tagged by
workflow, `--data-filter` can restrict a training run to one workflow's examples to
produce a second adapter without duplicating the base model.

## Validation metrics

`eval_on_start=True` evaluates the initial policy at **step 0**, before any optimizer
update. It logs both completion-only validation loss and the usual generation metrics
on the same validation conversations used later in training, including when MLflow is enabled.
This baseline is a measurement, not a saved candidate for best-checkpoint selection.

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
real `model.generate()` (greedy) on **all validation conversations** by default
(`--metrics-eval-samples 0`). A positive value explicitly selects a fixed subset for
smoke tests. The local validation split has 529 conversations, 651 assistant turns,
and 32 multi-turn conversations. Full generation evaluation takes substantially longer
than the old 50-conversation sample. Logs the full `metrics.py` breakdown each eval round (tool-selection
accuracy, param value/type accuracy, hallucination rate, per-error-type rates), not just
the single `eval_fc_call_correctness` scalar `metric_for_best_model` uses — useful for
telling *which* kind of error is driving a change in that scalar rather than just that
one happened.

Each evaluation saves `validation_predictions/step-NNNNNN.json` inside the output
directory, with conversation IDs, per-turn contexts, expected calls, predictions,
scores, generated-token counts, token-limit flags, and dataset/scorer/template hashes.
The report is also uploaded to MLflow immediately when `--mlflow` is enabled.
`eval_n_conversations` and `eval_n_trajectories` expose the sample sizes on the dashboard.

Every candidate is compared with its stage's step-zero baseline using 10,000 paired
bootstrap draws (seed 0), resampling whole conversations with replacement. Reports
contain candidate-minus-baseline differences and percentile 95% confidence intervals
for full-call, refusal, and trajectory accuracy. Deltas and interval bounds are also
logged as `eval_vs_baseline_*` metrics. Call/refusal rates are weighted by turns;
trajectory rates are weighted by conversations. Replicates with no eligible cases
are omitted for that metric and their valid count is reported.

These are pointwise intervals, not adjusted for selecting the best of many checkpoints.
Trajectory accuracy still uses ground-truth history, not a live agent rollout. Use
validation for checkpoint selection and reserve `test.jsonl` for the final comparison;
training callbacks only read the supplied validation file.

Compare any two saved reports without generating again (from the repository root):

```bash
python 2_evaluations/compare_validation.py \
  --baseline checkpoints/sft/validation_predictions/step-000000.json \
  --candidate checkpoints/sft/validation_predictions/step-000140.json
```

It rejects mismatched datasets, scorers, generation settings, or conversation/turn alignment.

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
  --metrics-eval-samples 0 \
  --mlflow --mlflow-experiment-name fin-agent-sft
```

This checkpoint is the starting policy for `../2_grpo/`.

## Tests

Run `tox -e sft-tests`. These CPU-only tests cover JSON loading, message conversion,
per-turn expansion, history isolation, call-type filtering, native call preservation,
no-call targets, non-thinking prompt boundaries, and invalid/empty inputs. They also
exercise TRL's real preprocessing and collator to verify prompt/padding masks and a
tiny randomly initialized Qwen model to verify step-zero evaluation before training.
No pretrained model download is required.
