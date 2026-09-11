# 1. Training

Two stages, run in order:

- **[`1_sft/`](1_sft/)** — supervised fine-tuning on the prepared ToolACE splits. This is
  the initial, simple training loop: one forward/backward pass per batch, teaching the
  base model the schema and format. Run this first.
- **[`2_grpo/`](2_grpo/)** — GRPO reinforcement learning on top of the SFT checkpoint,
  with a reward function that mirrors `2_evaluations/run_internal_eval.py`'s scoring —
  optimizing directly toward the metric that gates promotion, rather than token-level
  match against one reference completion. Staged *after* SFT, not instead of it: RL
  needs a policy that already produces mostly-valid completions to get useful reward
  signal, and it's a real step up in infra (rollout generation every step, not just a
  forward/backward pass).

Both stages share the base model requirement: it must match
`2_evaluations/run_bfcl_eval.py`'s baseline model (`Qwen/Qwen3-8B` by default) — a
baseline-vs-fine-tuned comparison only means something if every measurement traces back
to the same starting checkpoint. See `1_sft/README.md` for why Qwen3-8B specifically, and
the `enable_thinking` notes in both subfolders' READMEs and in `2_evaluations/README.md`
for how its hybrid-reasoning mode is handled at training and inference time.
