# configs/ — portable Kubernetes setup

Runs this project's jobs on a barebones single-GPU Linux VM (developed against 1×H100,
but anything with a recent NVIDIA GPU + driver works) — no dependency on any specific
existing cluster, node labels, or hostnames. If you already have a cluster with GPU
scheduling and MLflow running, skip `setup/` and go straight to `templates/`, pointing
`--mlflow-tracking-uri` at your own instance.

## Layout

```
configs/
├── setup/           # cluster bootstrap — k3s, GPU scheduling, MLflow. Run once.
└── templates/        # the actual workloads, grouped by what they do
    ├── training/      # fine-tuning jobs (SFT, GRPO)
    └── inference/      # model-serving deployments (vLLM) + benchmarks that call one
```

| File | Purpose |
|---|---|
| `setup/setup.sh` | Installs k3s, the NVIDIA runtime + device plugin (if a GPU is present), the `fin-agent` namespace, MLflow, and every job source ConfigMap (via `rebuild-configmaps.sh`). Run this first. |
| `setup/rebuild-configmaps.sh` | Rebuilds every job source ConfigMap from the current local checkout. **Run this after every `git pull`** — a Job's ConfigMap is a separate, already-applied object nothing updates automatically, so a stale one silently keeps running old code with no error. `setup.sh` calls this same script; it's also here standalone so a pull doesn't require rerunning all of `setup.sh`. |
| `setup/nvidia-runtimeclass.yaml`, `setup/nvidia-device-plugin.yaml` | Let Kubernetes schedule against the GPU (`nvidia.com/gpu`). Applied by `setup.sh`; no time-slicing — one GPU, one workload at a time, matching how the templates below are designed to be run. |
| `setup/namespace.yaml` | The `fin-agent` namespace everything else lives in. |
| `setup/mlflow.yaml` | Self-contained MLflow (ClusterIP + SQLite + hostPath) — a fresh instance for this VM, not tied to any external ingress hostname. |
| `templates/training/prepare-dataset-job.yaml` | Runs `0_data/prepare_dataset.py` once, writing train/val/test.jsonl to a shared PVC (`fin-agent-dataset`). Run this before any other training job below — they all read from it. |
| `templates/training/sft-smoke-job.yaml` | SFT smoke test — 10 training steps, a step-based eval, and a checkpoint save. |
| `templates/training/sft-job.yaml` | The real SFT run — full ToolACE-derived dataset, `train_sft.py`'s own defaults (1 epoch, gradient checkpointing, eval/save every 140 steps), checkpoint persisted to hostPath. |
| `templates/training/grpo-smoke-job.yaml` | GRPO smoke test — dataset construction, reward function, rollout generation, a held-out eval pass (same `FunctionCallEvalCallback` SFT uses, imported from `train_sft.py` — see `1_training/2_grpo/README.md`), and a checkpoint save. Warm-starts from `sft-smoke-job.yaml`'s checkpoint, not the raw base model. |
| `templates/training/run-grpo-smoke-chain.sh` | Runs dataset prep → SFT smoke test → GRPO smoke test in sequence, one command — exercises the SFT→GRPO checkpoint handoff end-to-end. |
| `templates/training/grpo-job.yaml` | The real GRPO run — full dataset, `train_grpo.py`'s own defaults (held-out eval every 140 steps, `load_best_model_at_end` on `fc_call_correctness` — no longer just "whatever the final step produced"), warm-started from `sft-job.yaml`'s checkpoint (not the smoke one). Run `grpo-smoke-job.yaml` first to prove the pipeline works. |
| `templates/training/run-sft-grpo-chain.sh` | Runs dataset prep → real SFT run → real GRPO run in sequence, one command — the same idea as `run-grpo-smoke-chain.sh`, applied to the two real (hours-to-days) runs. Run the smoke chain first. |
| `templates/inference/vllm-qwen3-8b.yaml` | vLLM serving the Qwen3-8B baseline — a standing Deployment, not a one-off Job. |
| `templates/inference/vllm-grpo-mtp.yaml` | vLLM serving a specific MLflow-tracked run's model artifact (substitute `__RUN_ID__` first — `sed 's/__RUN_ID__/<run id>/g' vllm-grpo-mtp.yaml \| kubectl apply -f -`), pinning what's served to a known tracked run instead of whatever's currently on the checkpoints hostPath. An initContainer pulls it via `mlflow.artifacts.download_artifacts` from the same `fin-agent-mlflow-artifacts` PVC `setup/mlflow.yaml` writes to. Has MTP speculative decoding (4 candidate tokens) and structured-output (guided) decoding enabled — MTP needs a checkpoint with a trained MTP head, which this project's plain dense Qwen3-8B fine-tune does not have, so drop `--speculative-config` unless the run you point at actually has one. |
| `templates/inference/bfcl-eval-job.yaml` | BFCL evaluation against the vLLM deployment above. **Depends on it being up** — use `run-bfcl-eval.sh`, not a bare `kubectl apply`, unless you're managing that dependency yourself. |
| `templates/inference/run-bfcl-eval.sh` | Applies the vLLM deployment, waits for it to be Ready, then applies and follows the eval Job — the one-command way to run the benchmark without forgetting the server it needs. |
| `templates/inference/bfcl-eval-checkpoint-job.yaml` | Template (substitute `__CHECKPOINT__` — see its header) for a BFCL evaluation against a local training checkpoint, with its own bundled vLLM Deployment. Defaults to one small category (`live_relevance`) for a quick, cheap check. Rendered and applied by `run-bfcl-eval-suite.sh`, not `kubectl apply`ed directly. |
| `templates/inference/run-bfcl-eval-suite.sh` | Evaluates the raw baseline plus every local checkpoint (smoke-test/sft/grpo) that exists on this node against BFCL, one at a time, in one command — smoke scale by default (`FULL_SCALE=1` for the real `python` category). Logs each model's `bfcl_non_live_ast_accuracy`/`bfcl_live_ast_accuracy` to its own MLflow experiment, alongside the public leaderboard's reference numbers for direct comparison. |

## Quickstart

```bash
./setup/setup.sh

# from the fin_agent project root — prepares train/val/test.jsonl (+ smoke-sized
# slices) once, onto a PVC every training job below reads from:
kubectl create configmap fin-agent-data-prep-src -n fin-agent \
  --from-file=prepare_dataset.py=0_data/prepare_dataset.py
kubectl apply -f configs/templates/training/prepare-dataset-job.yaml
kubectl -n fin-agent wait --for=condition=complete job/fin-agent-prepare-dataset --timeout=1800s

kubectl create configmap fin-agent-sft-src -n fin-agent \
  --from-file=prepare_dataset.py=0_data/prepare_dataset.py \
  --from-file=run_internal_eval.py=2_evaluations/run_internal_eval.py \
  --from-file=metrics.py=1_training/1_sft/metrics.py \
  --from-file=train_sft.py=1_training/1_sft/train_sft.py \
  --from-file=tox.ini=tox.ini

kubectl apply -f configs/templates/training/sft-smoke-job.yaml
kubectl -n fin-agent logs -l app=fin-agent-sft-smoke -f
```

Once the smoke test passes, the real run reuses the same `fin-agent-sft-src` ConfigMap
and the same prepared-dataset PVC (no separate ConfigMap or re-prep needed):

```bash
kubectl apply -f configs/templates/training/sft-job.yaml
kubectl -n fin-agent logs -l app=fin-agent-sft -f
```

Or run either chain (smoke or real) end-to-end in one command — see
`templates/training/run-grpo-smoke-chain.sh` / `run-sft-grpo-chain.sh`.

For the BFCL benchmark, build its ConfigMap (command in
`templates/inference/bfcl-eval-job.yaml`'s header) then run
`./templates/inference/run-bfcl-eval.sh` — it takes care of bringing up the vLLM
deployment first. `templates/training/grpo-smoke-job.yaml` follows the same
ConfigMap-then-apply pattern as the SFT smoke test (and reads the same prepared-dataset
PVC); see its own header.

## Design notes (why this looks the way it does)

- **One GPU, one workload.** Nothing under `templates/` runs concurrently with anything
  else there by design — every manifest's header says so. If you have more than one GPU,
  drop the `nvidia.com/gpu: 1` assumption and adjust; these don't attempt multi-GPU
  scheduling.
- **Benchmarks live under `inference/`, not their own directory.** `bfcl-eval-job.yaml`
  doesn't bring up its own vLLM server — it only polls for one — so it's grouped with the
  `vllm-qwen3-8b.yaml` deployment it depends on rather than off on its own. That
  deployment stays reusable beyond just this one benchmark (point anything else that
  wants an OpenAI-compatible endpoint at it), but applying the eval Job alone still does
  nothing useful without the server already running — hence `run-bfcl-eval.sh` as the
  actual entry point for that benchmark.
- **Dataset preparation is a job of its own, not baked into every training job.**
  ToolACE-derived train/val/test splits run well past the ~1MiB ConfigMap size limit, so
  `prepare-dataset-job.yaml` runs `0_data/prepare_dataset.py` once (deterministic,
  seeded — reruns produce byte-identical splits) onto a shared PVC that every training
  job (`sft-smoke-job.yaml`, `sft-job.yaml`, `grpo-smoke-job.yaml`, `grpo-job.yaml`)
  mounts read-only. The smoke-sized slices (`smoke_train.jsonl`, `smoke_val.jsonl`) are
  cut once there too, rather than each smoke job re-slicing the full files on every run.
- **No node labels, no hostname pinning.** Every `nodeSelector` that referenced a
  specific machine name was removed — Kubernetes schedules purely on the
  `nvidia.com/gpu` resource request, which is what makes this "usable anywhere" rather
  than tied to one cluster's naming scheme.
- **MLflow's Host/CORS checks are wildcarded** (`setup/mlflow.yaml`) rather than pinned
  to a specific hostname or ClusterIP, because there isn't one to pin to ahead of time on
  an arbitrary VM. This was a real, hard-won fix during development — mlflow's own web UI
  403s on its `/ajax-api/*` calls without `MLFLOW_SERVER_CORS_ALLOWED_ORIGINS`, separately
  from the `MLFLOW_SERVER_ALLOWED_HOSTS` check for the Host header itself. Tighten both
  if you expose this MLflow instance beyond this one VM.
- **Code ships as ConfigMaps, not a custom image.** Every job pip-installs its
  dependencies at pod start from a stock image (`nvcr.io/nvidia/pytorch`, `python:slim`,
  or the official `vllm/vllm-openai` image). Slower to start than a prebuilt image would
  be, but means nothing here depends on a container registry you'd need to push to
  first — genuinely "clone and run."
- **Image tags matter more than they look.** `nvcr.io/nvidia/pytorch:26.05-py3` was
  chosen for broad CUDA compute-capability coverage; an older NGC tag can silently ship a
  torch build that doesn't recognize a newer GPU architecture at all (hit this for real
  during development — a stale tag loaded fine, then failed with "GPU with CUDA
  capability sm_XX is not compatible with the current PyTorch installation" the moment
  training actually touched the GPU). If you hit that error, try a newer tag first.
