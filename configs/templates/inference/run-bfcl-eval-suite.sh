#!/usr/bin/env bash
# Evaluates a set of models against BFCL in one command: the raw baseline
# (Qwen/Qwen3-8B, no fine-tuning) plus every local checkpoint listed in CHECKPOINTS
# below that actually exists on this node -- the same "run several stages without
# hand-typing the order and forgetting to wait between them" idea as
# ../training/run-grpo-smoke-chain.sh, applied to evaluation instead of training.
#
# Defaults to smoke scale for every model (--test-category live_relevance, 18 cases,
# minutes not hours) so the whole suite is cheap to run repeatedly while iterating --
# see FULL_SCALE below to switch every entry over to the real, leaderboard-comparable
# `python` category (3491 cases, hours) once specific checkpoints are worth that cost.
#
# Usage:
#   ./run-bfcl-eval-suite.sh              # smoke scale (default)
#   FULL_SCALE=1 ./run-bfcl-eval-suite.sh # full python category for every model
#
# Each stage's Job/Deployment is deleted and reapplied if a prior one exists: Job pod
# templates are immutable, so a bare `kubectl apply` on an already-existing Job
# silently keeps using its OLD template instead of erroring or updating -- see
# rebuild-configmaps.sh's header for the real debugging time this already cost once in
# this project. Only one model is served at a time (this project targets a single GPU
# running one workload at a time): each model's Deployment+Job is torn down before the
# next one is applied, not left running alongside it.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE=fin-agent
TEST_CATEGORY="live_relevance"
if [ "${FULL_SCALE:-0}" = "1" ]; then
  TEST_CATEGORY="python"
  echo ">> FULL_SCALE=1: evaluating every model on the full 'python' category (hours per model, not minutes)"
fi

# Names double as the hostPath subdirectory under /var/lib/fin-agent/checkpoints:
#   smoke-test -> ../training/smoke-test-job.yaml (job fin-agent-sft-smoke-test)
#   sft        -> ../training/sft-job.yaml (job fin-agent-sft)
#   grpo       -> ../training/grpo-real-job.yaml (job fin-agent-grpo)
# A plain array (not an associative one) so this only needs bash 3.2+, not bash 4 --
# unverified which the target VM ships. Local, single-node filesystem checks below (this
# script runs ON the same VM the checkpoints live on, same as setup.sh) skip whichever
# entries don't exist yet rather than failing the whole suite over one missing checkpoint.
CHECKPOINTS=(smoke-test sft grpo)

run_baseline() {
  echo ">> [baseline] Qwen/Qwen3-8B (no fine-tuning)"
  kubectl -n "${NAMESPACE}" delete job fin-agent-bfcl-eval --ignore-not-found --wait=true
  kubectl -n "${NAMESPACE}" delete deployment fin-agent-vllm-qwen3-8b --ignore-not-found --wait=true
  kubectl apply -f "${SCRIPT_DIR}/vllm-qwen3-8b.yaml"
  kubectl -n "${NAMESPACE}" rollout status deploy/fin-agent-vllm-qwen3-8b --timeout=900s
  sed "s/--test-category python/--test-category ${TEST_CATEGORY}/" "${SCRIPT_DIR}/bfcl-eval-job.yaml" \
    | kubectl apply -f -
  kubectl -n "${NAMESPACE}" wait --for=condition=complete job/fin-agent-bfcl-eval \
    --timeout="$([ "${TEST_CATEGORY}" = "python" ] && echo 21600s || echo 1800s)"
  kubectl -n "${NAMESPACE}" logs -l app=fin-agent-bfcl-eval --tail=20
  kubectl -n "${NAMESPACE}" delete -f "${SCRIPT_DIR}/vllm-qwen3-8b.yaml" --ignore-not-found
}

run_checkpoint() {
  local name="$1"
  echo ">> [${name}] /var/lib/fin-agent/checkpoints/${name}"
  if [ ! -d "/var/lib/fin-agent/checkpoints/${name}" ]; then
    echo "   skipping: no checkpoint at /var/lib/fin-agent/checkpoints/${name} yet (see CHECKPOINTS in this script for which job produces it)"
    return
  fi
  local rendered
  rendered="$(sed -e "s/__CHECKPOINT__/${name}/g" -e "s/--test-category live_relevance/--test-category ${TEST_CATEGORY}/" \
    "${SCRIPT_DIR}/bfcl-eval-checkpoint-job.yaml")"
  kubectl -n "${NAMESPACE}" delete job "fin-agent-bfcl-eval-${name}" --ignore-not-found --wait=true
  kubectl -n "${NAMESPACE}" delete deployment "fin-agent-vllm-${name}" --ignore-not-found --wait=true
  echo "${rendered}" | kubectl apply -f -
  kubectl -n "${NAMESPACE}" rollout status "deploy/fin-agent-vllm-${name}" --timeout=600s
  kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/fin-agent-bfcl-eval-${name}" \
    --timeout="$([ "${TEST_CATEGORY}" = "python" ] && echo 21600s || echo 1800s)"
  kubectl -n "${NAMESPACE}" logs -l "app=fin-agent-bfcl-eval-${name}" --tail=20
  echo "${rendered}" | kubectl delete -f - --ignore-not-found
}

run_baseline
for name in "${CHECKPOINTS[@]}"; do
  run_checkpoint "${name}"
done

echo ""
echo "=============================================================="
echo " Suite complete. Compare results in MLflow:"
echo "   kubectl -n ${NAMESPACE} port-forward svc/mlflow 5000:5000"
echo "   then open http://localhost:5000 -- baseline logs to experiment"
echo "   'fin-agent-bfcl', each checkpoint to its own"
echo "   'fin-agent-bfcl-eval-<name>' experiment. bfcl_non_live_ast_accuracy /"
echo "   bfcl_live_ast_accuracy sit alongside leaderboard_non_live_ast_accuracy /"
echo "   leaderboard_live_ast_accuracy in every run for direct comparison."
echo "=============================================================="
