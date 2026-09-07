#!/usr/bin/env bash
# Evaluates three specific models against BFCL in one command, each showing both
# Non-Live and Live AST accuracy: the raw baseline (Qwen/Qwen3-8B, no fine-tuning) plus
# two MLflow-tracked checkpoints pinned by run_id (SFT and GRPO below) — the run_id
# analogue of ../training/run-bfcl-eval-suite.sh, which instead sweeps whatever local
# hostPath checkpoints happen to exist. Use this one when you want to evaluate specific,
# known runs (e.g. to compare against a particular MLflow experiment entry) rather than
# "whatever's currently on disk."
#
# Defaults to smoke scale (--test-category-set below: "parallel live_parallel", 199+15=
# 214 cases) so the whole suite is cheap to run repeatedly while iterating -- see
# FULL_SCALE to switch every entry over to the real, leaderboard-comparable `python`
# category (3491 cases, hours) once specific runs are worth that cost. Both scales
# populate bfcl_non_live_ast_accuracy AND bfcl_live_ast_accuracy for every model: smoke
# does it by running one small Non-Live category (parallel, 199 cases) and one small
# Live category (live_parallel, 15 cases) in the same MLflow run (see
# bfcl-eval-mlflow-checkpoint-job.yaml's header for the mechanism -- no small built-in
# BFCL collection spans both groups on its own), and the real `python` collection
# already spans both groups in one category.
#
# Usage:
#   ./run-bfcl-eval-run-ids-suite.sh              # smoke scale (default)
#   FULL_SCALE=1 ./run-bfcl-eval-run-ids-suite.sh # full python category for every model
#   SFT_RUN_ID=<id> GRPO_RUN_ID=<id> ./run-bfcl-eval-run-ids-suite.sh   # override the run_ids below
#
# Each stage's Job/Deployment is deleted and reapplied if a prior one exists -- same
# reasoning as run-bfcl-eval-suite.sh (Job pod templates are immutable, a bare
# `kubectl apply` on an already-existing Job silently keeps its OLD template). Only one
# model is served at a time (this project targets a single GPU running one workload at a
# time): each model's Deployment+Job is torn down before the next one is applied.
#
# Prerequisites:
#   1. setup/setup.sh (creates the namespace + MLflow this suite logs to and, for the
#      two checkpoints, pulls the model artifact from)
#   2. The fin-agent-bfcl-src ConfigMap (see bfcl-eval-job.yaml's header for the build
#      command, or rerun setup/rebuild-configmaps.sh)
#   3. Both MLflow runs below need a "model" artifact already logged (train_sft.py/
#      train_grpo.py with --mlflow both log one) -- these are NOT produced by this
#      script, only consumed.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE=fin-agent

# Defaults are the two run_ids this suite was built for; override via env var to point
# at different runs without editing this file.
SFT_RUN_ID="${SFT_RUN_ID:-e783b52f2b7a42cc8ff2b2786949cb71}"
GRPO_RUN_ID="${GRPO_RUN_ID:-7ed9c74413964bd4b69ba4e405ad2060}"

TEST_CATEGORIES="parallel live_parallel"
WAIT_TIMEOUT=1800s
if [ "${FULL_SCALE:-0}" = "1" ]; then
  TEST_CATEGORIES="python"
  WAIT_TIMEOUT=21600s
  echo ">> FULL_SCALE=1: evaluating every model on the full 'python' category (hours per model, not minutes)"
fi

run_baseline() {
  echo ">> [baseline] Qwen/Qwen3-8B (no fine-tuning)"
  kubectl -n "${NAMESPACE}" delete job fin-agent-bfcl-eval-qwen3-8b --ignore-not-found --wait=true
  kubectl -n "${NAMESPACE}" delete deployment fin-agent-vllm-qwen3-8b --ignore-not-found --wait=true
  kubectl apply -f "${SCRIPT_DIR}/vllm-qwen3-8b.yaml"
  kubectl -n "${NAMESPACE}" rollout status deploy/fin-agent-vllm-qwen3-8b --timeout=900s
  sed "s/__TEST_CATEGORIES__/${TEST_CATEGORIES}/g" "${SCRIPT_DIR}/bfcl-eval-baseline-categories-job.yaml" \
    | kubectl apply -f -
  kubectl -n "${NAMESPACE}" wait --for=condition=complete job/fin-agent-bfcl-eval-qwen3-8b --timeout="${WAIT_TIMEOUT}"
  kubectl -n "${NAMESPACE}" logs -l app=fin-agent-bfcl-eval-qwen3-8b --tail=20
  kubectl -n "${NAMESPACE}" delete -f "${SCRIPT_DIR}/vllm-qwen3-8b.yaml" --ignore-not-found
}

run_mlflow_checkpoint() {
  local name="$1" run_id="$2"
  echo ">> [${name}] MLflow run_id=${run_id}"
  local rendered
  rendered="$(sed -e "s/__NAME__/${name}/g" -e "s/__RUN_ID__/${run_id}/g" -e "s/__TEST_CATEGORIES__/${TEST_CATEGORIES}/g" \
    "${SCRIPT_DIR}/bfcl-eval-mlflow-checkpoint-job.yaml")"
  kubectl -n "${NAMESPACE}" delete job "fin-agent-bfcl-eval-${name}" --ignore-not-found --wait=true
  kubectl -n "${NAMESPACE}" delete deployment "fin-agent-vllm-${name}" --ignore-not-found --wait=true
  echo "${rendered}" | kubectl apply -f -
  kubectl -n "${NAMESPACE}" rollout status "deploy/fin-agent-vllm-${name}" --timeout=900s
  kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/fin-agent-bfcl-eval-${name}" --timeout="${WAIT_TIMEOUT}"
  kubectl -n "${NAMESPACE}" logs -l "app=fin-agent-bfcl-eval-${name}" --tail=20
  echo "${rendered}" | kubectl delete -f - --ignore-not-found
}

run_baseline
run_mlflow_checkpoint sft "${SFT_RUN_ID}"
run_mlflow_checkpoint grpo "${GRPO_RUN_ID}"

echo ""
echo "=============================================================="
echo " Suite complete. Compare results in MLflow:"
echo "   kubectl -n ${NAMESPACE} port-forward svc/mlflow 5000:5000"
echo "   then open http://localhost:5000 -- baseline logs to experiment"
echo "   'fin-agent-bfcl', sft/grpo to their own"
echo "   'fin-agent-bfcl-eval-<name>' experiments. Every run has both"
echo "   bfcl_non_live_ast_accuracy and bfcl_live_ast_accuracy (plus the"
echo "   leaderboard_* reference numbers) for direct comparison across"
echo "   all three."
echo "=============================================================="
