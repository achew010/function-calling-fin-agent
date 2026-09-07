#!/usr/bin/env bash
# Rebuilds every job source ConfigMap this project uses, from the current state of the
# local checkout. Run this after every `git pull` (or any local edit to one of the
# .py/tox.ini files below) and before applying/reapplying a Job -- otherwise it silently
# keeps running whatever code was baked into the ConfigMap last time. This is the exact,
# repeated failure mode that cost real debugging time across this project: `git pull`
# updates the local files, but a Job's ConfigMap is a separate, already-applied object
# that nothing updates automatically, and a stale ConfigMap produces no error or
# warning -- just an old version of the code running silently, looking like a real bug.
#
# setup.sh calls this same script during initial cluster bootstrap; this file exists
# standalone too so it can be rerun on its own, without repeating the rest of setup.sh's
# (idempotent but heavier -- GPU/k3s checks, MLflow rollout) cluster bootstrap work.
#
# Usage:
#   ./rebuild-configmaps.sh
#
# Verify a specific rebuild actually took before spending a run on it, e.g.:
#   kubectl get configmap fin-agent-grpo-src -n fin-agent -o jsonpath='{.data.train_grpo\.py}' | grep -c merge_and_unload

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NAMESPACE=fin-agent

echo ">> Rebuilding job source ConfigMaps from ${PROJECT_ROOT}"

kubectl create configmap fin-agent-data-prep-src -n "${NAMESPACE}" \
  --from-file=prepare_dataset.py="${PROJECT_ROOT}/0_data/prepare_dataset.py" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap fin-agent-sft-src -n "${NAMESPACE}" \
  --from-file=prepare_dataset.py="${PROJECT_ROOT}/0_data/prepare_dataset.py" \
  --from-file=run_internal_eval.py="${PROJECT_ROOT}/2_evaluations/run_internal_eval.py" \
  --from-file=metrics.py="${PROJECT_ROOT}/1_training/1_sft/metrics.py" \
  --from-file=train_sft.py="${PROJECT_ROOT}/1_training/1_sft/train_sft.py" \
  --from-file=tox.ini="${PROJECT_ROOT}/tox.ini" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap fin-agent-grpo-src -n "${NAMESPACE}" \
  --from-file=prepare_dataset.py="${PROJECT_ROOT}/0_data/prepare_dataset.py" \
  --from-file=run_internal_eval.py="${PROJECT_ROOT}/2_evaluations/run_internal_eval.py" \
  --from-file=metrics.py="${PROJECT_ROOT}/1_training/1_sft/metrics.py" \
  --from-file=train_sft.py="${PROJECT_ROOT}/1_training/1_sft/train_sft.py" \
  --from-file=train_grpo.py="${PROJECT_ROOT}/1_training/2_grpo/train_grpo.py" \
  --from-file=tox.ini="${PROJECT_ROOT}/tox.ini" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap fin-agent-bfcl-src -n "${NAMESPACE}" \
  --from-file=run_bfcl_eval.py="${PROJECT_ROOT}/2_evaluations/run_bfcl_eval.py" \
  --from-file=log_bfcl_to_mlflow.py="${PROJECT_ROOT}/2_evaluations/log_bfcl_to_mlflow.py" \
  --dry-run=client -o yaml | kubectl apply -f -

echo ">> Done: fin-agent-data-prep-src, fin-agent-sft-src, fin-agent-grpo-src, fin-agent-bfcl-src"
