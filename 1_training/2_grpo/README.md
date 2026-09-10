# 2. GRPO

Generation-based checkpoint evaluation uses the full validation split by default
(`--metrics-eval-samples 0`), including a step-zero measurement of the starting SFT
policy. The shared SFT callback saves per-turn predictions and paired conversation
bootstrap comparisons to this baseline; see `../1_sft/README.md` for report paths and
comparison commands. Positive sample counts remain available for smoke tests.
Reserve the test split for the final comparison after selecting a checkpoint.

**Why this exists as a separate, later stage — not the initial training loop:** function
calling has a rare property that makes RL genuinely attractive: the reward is trivially,
deterministically computable (parse the output, check the function name and arguments
against ground truth). That's almost exactly what `2_evaluations/run_internal_eval.py`
already does — so GRPO here is a way to optimize *directly* toward the metric that gates
promotion, instead of token-level match against ToolACE's one reference completion, and
to encode "some mistakes are worse than others" (hallucination vs. a minor formatting
miss) directly into the objective, which SFT has no way to express.

It's staged *after* `../1_sft/`, not instead of it, for concrete reasons:
- RL needs a policy that already produces a reasonable fraction of valid completions —
  starting GRPO from an untuned base model gives a near-zero, uninformative reward
  signal almost everywhere. SFT-then-RL is the standard pattern (this mirrors DeepSeek-R1's
  own "cold-start SFT, then RL" recipe), not RL from scratch.
- It's a real infra step up from SFT: rollout generation (sampling `num_generations`
  completions per prompt from the *current* policy, every step) instead of one
  forward/backward pass — meaningfully more compute per step, and something to budget
  GPU time for separately from serving.
- A buggy reward function gets *exploited*, not just ignored — GRPO will find and climb
  any gap in `compute_reward` (e.g. under-checking argument values). That demands the
  same adversarial scrutiny this project already applied to the eval's parsing edge
  cases (see `0_data/README.md`'s verification section), but now a miss teaches the
  model the wrong thing rather than just mis-scoring an eval run.

## Reward design

`compute_reward` (in `train_grpo.py`) mirrors `run_internal_eval.py`'s scoring
dimensions, collapsed into one scalar with partial credit:

| Case | Reward |
|---|---|
| Correct refusal (no call expected, none made) | `+1.0` |
| Unwarranted call (no call expected, one made) | `-1.0` |
| Missed call (call expected, none made) | `-1.0` |
| Hallucinated call (function not in the tool list) | `-2.0` — worst case |
| Wrong function entirely | `-0.5` |
| Right function, wrong arguments | `+0.3` |
| Right function, right arguments | `+1.0` |

This is a **starting design, not a final one** — the natural extension, once
`0_data/prepare_dataset.py`'s `risk_tier` field carries real values instead of
`"unclassified"`, is to scale the hallucination/wrong-function penalties by risk tier
(hallucinating a high-value transfer function should cost far more than hallucinating a
low-risk lookup) — directly wiring the business brief's risk-tiering into the training
objective, not just the eval gate.

## Held-out eval

Previously GRPO had none: the only correctness signal was `reward_func`, computed on
rollouts sampled from the *training* set — an optimization target, not a generalization
check, and with no "best" checkpoint to fall back on if the policy started drifting, just
whatever the final step produced.

Now `train_grpo.py` imports `FunctionCallEvalCallback` directly from `../1_sft/train_sft.py`
(not reimplemented — same reasoning as reusing `run_internal_eval.py`: the eval mechanism
and the metric name it reports, `fc_call_correctness`, should be identical across both
stages, not two copies that can quietly drift apart) and wires it up the same way SFT
does: `--val-file` (now required), `--eval-strategy`/`--eval-steps` (default `steps`/140),
`load_best_model_at_end=True` with `metric_for_best_model="fc_call_correctness"`. At the
end of `trainer.train()`, whichever checkpoint had the best real, generation-based
`fc_call_correctness` gets reloaded, merged, and saved — not necessarily the final step's.

Separately, passing `eval_dataset` to `GRPOTrainer` also turns on its own built-in
generation-based eval loop (reward/KL/entropy/completion-length, computed the same way as
training rollouts) on the held-out split, logged alongside `FunctionCallEvalCallback`'s
richer `metrics.py` breakdown — real cost, since it's full generation over the whole eval
set, not the cheap teacher-forced forward pass SFT's analogous eval loop uses; `--eval-batch-size`
and `num_generations_eval` (TRL's own `GRPOConfig` field) are the knobs if that turns out
to be too expensive for a given val split size.

## Code structure

- **`build_grpo_dataset`** — reuses `run_internal_eval.py`'s `build_eval_instances`
  (imported, not reimplemented): one row per assistant turn, same instances the internal
  eval scores checkpoints against. Columns: `prompt` (conversational messages, no
  target — GRPO generates it), `tool_names`, `is_call_case`, `expected_calls`.
  `expected_calls` is stored JSON-encoded rather than as a nested column — argument
  values are heterogeneous (str/int/float/list/dict) across examples, which breaks
  Arrow's struct-column type inference (`Dataset.from_dict` raised `ArrowInvalid:
  cannot mix struct and non-struct, non-null values` on the raw nested form — caught by
  actually running this against the real prepared data, not assumed).
- **`reward_func`** — TRL's `GRPOTrainer` reward-function contract: `completions` is one
  message per generation, every other dataset column is passed through as a
  same-length aligned list (verified against the installed `trl` version's
  `GRPOTrainer`/`trl.rewards` source, not assumed — the "extra columns become reward
  kwargs" behavior is exactly how `trl.rewards.think_format_reward` is written).
- **`enable_thinking=False`** via `GRPOConfig.chat_template_kwargs` — same non-thinking
  requirement as everywhere else in this project, so rollouts during training match how
  the model is actually served.

Verified without a GPU: `build_grpo_dataset` runs end-to-end against the real
`0_data/data/*.jsonl` splits, and `compute_reward`/`reward_func` were self-consistency
checked the same way `0_data/README.md` checked the call-syntax parser — feeding a
"perfect" prediction (the exact ground-truth call, or a plausible refusal) back through
`compute_reward` for every instance in `test.jsonl`: **all 648 score the maximum reward
(1.0)**, and the hallucination/wrong-function/wrong-argument cases score `-2.0`/`-0.5`/
`0.3` exactly as designed. `GRPOConfig`/`LoraConfig` construct cleanly with the flags
this script passes. Not verified here: an actual training run — that needs a GPU to load
`Qwen/Qwen3-8B` (or the SFT checkpoint) and run rollout generation, which this
environment doesn't have.

## Run

```bash
python train_grpo.py \
  --base-model ../1_sft/checkpoints/adapter-general \
  --train-file ../../0_data/data/train.jsonl \
  --val-file ../../0_data/data/val.jsonl \
  --output-dir checkpoints/grpo-general \
  --num-generations 8 --lr 1e-6 --epochs 1
```

## Rollout generation and batch sizing

Rollout generation is what makes a GRPO step expensive: `--num-generations` (8)
completions per prompt, each up to `--max-completion-length` (512) tokens, decoded
autoregressively *every step*, through the policy model's own `.generate()`
(`GRPOConfig.use_vllm` stays at its default of `False`). Routing that through vLLM was
tried and reverted — colocate mode needs `vllm` importable in the training process, and
installing it alongside the NGC image's CUDA stack breaks `peft`'s import-time
`transformer_engine` probe (`undefined symbol: cublasLtGroupedMatrixLayoutInit_internal`)
before training starts.

**`--per-device-batch-size` cannot be lowered on its own.** TRL derives
`generation_batch_size = per_device_batch_size × world_size × grad_accum` and rejects any
value not divisible by `--num-generations`, because a generation batch has to contain
whole prompt groups (verified against the installed `trl`'s own `GRPOConfig.__post_init__`,
not assumed). Hence the defaults: batch 4 with `--grad-accum 2` keeps
`generation_batch_size` at 8 = `G`, halving per-step activation memory without changing
the effective batch or the group size. Batch 4 with `--grad-accum 1` fails at startup.

The same rule applies to eval with its own group size: `--num-generations-eval` (2, well
below the training `G`) has to divide `--eval-batch-size` (4). Eval only needs completions
to score, not a group wide enough to estimate advantages from, so it also runs 4x less
generation than reusing `G=8` would.

`--mlflow`/`--mlflow-experiment-name`/`--mlflow-tracking-uri` and `--max-steps` mirror
`1_sft/train_sft.py`'s flags (same MLflowCallback wiring, same "override defaults via
CLI flags for a smoke run, never by editing the script" pattern).
