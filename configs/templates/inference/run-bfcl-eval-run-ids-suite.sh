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
#   QUANTIZATION=fp8_per_tensor FULL_SCALE=1 MODELS="baseline sft" \
#     SFT_RUN_ID=<id> ./run-bfcl-eval-run-ids-suite.sh   # same eval, weights quantized on load
#
# QUANTIZATION serves the SAME artifacts through vLLM's online (load-time) quantization,
# so an FP8 parity check needs no separate checkpoint. Results land under suffixed names
# (e.g. experiment fin-agent-bfcl-eval-sft-fp8-per-tensor, results /data/bfcl-sft-fp8-per-tensor)
# so they sit beside the bf16 numbers rather than overwriting the very thing they're being
# compared against. Run the bf16 pass first, then the quantized one, then compare
# bfcl_non_live_ast_accuracy/bfcl_live_ast_accuracy across the two experiments.
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
#
# Each leg's Job clears its own /data/bfcl-<name> results directory before running bfcl
# generate (see bfcl-eval-baseline-categories-job.yaml / bfcl-eval-mlflow-checkpoint-job.yaml)
# -- otherwise bfcl generate resumes from whatever result files are already there,
# silently skipping cases from a prior (possibly broken) run instead of regenerating them.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE=fin-agent

# Defaults are the two run_ids this suite was built for; override via env var to point
# at different runs without editing this file.
SFT_RUN_ID="${SFT_RUN_ID:-e783b52f2b7a42cc8ff2b2786949cb71}"
GRPO_RUN_ID="${GRPO_RUN_ID:-7ed9c74413964bd4b69ba4e405ad2060}"

# Which legs to run, in order. Trim this to evaluate a subset -- e.g.
# MODELS="baseline sft" before the GRPO checkpoint exists -- so an unwanted leg doesn't
# spend GPU time (or fail on a run_id whose model artifact isn't logged yet).
MODELS="${MODELS:-baseline sft grpo}"

TEST_CATEGORIES="parallel live_parallel"
WAIT_TIMEOUT=1800s
if [ "${FULL_SCALE:-0}" = "1" ]; then
  TEST_CATEGORIES="python"
  WAIT_TIMEOUT=21600s
  echo ">> FULL_SCALE=1: evaluating every model on the full 'python' category (hours per model, not minutes)"
fi

# QUANTIZATION runs the same evaluation against a vLLM server that quantizes the weights
# on load, e.g. QUANTIZATION=fp8_per_tensor (a real vLLM online-quantization shorthand --
# verified against vllm's own QuantizationMethods registry, which registers
# fp8_per_tensor/fp8_per_block/fp8_per_channel as online shorthands). No pre-quantized
# checkpoint is needed: the same bf16 artifact is quantized at load time.
#
# Every name a run touches gets VARIANT appended -- Deployment/Service, Job, the BFCL
# results directory and the MLflow experiment -- so a quantized run is a SEPARATE,
# side-by-side result rather than something that overwrites the bf16 numbers it exists to
# be compared against. Underscores are not legal in Kubernetes object names, hence the tr.
QUANTIZATION="${QUANTIZATION:-}"
VLLM_EXTRA_ARGS=""
VARIANT=""
if [ -n "${QUANTIZATION}" ]; then
  VLLM_EXTRA_ARGS="--quantization ${QUANTIZATION}"
  VARIANT="-$(printf '%s' "${QUANTIZATION}" | tr '_' '-')"
  echo ">> QUANTIZATION=${QUANTIZATION}: serving quantized, results suffixed '${VARIANT}'"
fi

run_baseline() {
  local name="qwen3-8b${VARIANT}"
  echo ">> [baseline] Qwen/Qwen3-8B (no fine-tuning)${VARIANT:+ [${QUANTIZATION}]}"
  kubectl -n "${NAMESPACE}" delete job "fin-agent-bfcl-eval-${name}" --ignore-not-found --wait=true
  kubectl -n "${NAMESPACE}" delete deployment "fin-agent-vllm-${name}" --ignore-not-found --wait=true
  # vllm-qwen3-8b.yaml is deliberately NOT a template -- it stays directly
  # kubectl-apply-able, as its own header documents and other manifests reference. So the
  # unquantized path applies it untouched (byte-identical to before), and only the
  # quantized path renders a copy. That render is verified rather than trusted: a sed
  # that silently matched nothing would serve bf16 while every name claimed FP8, which is
  # exactly the kind of silently-wrong result this suite has been bitten by before.
  if [ -z "${QUANTIZATION}" ]; then
    kubectl apply -f "${SCRIPT_DIR}/vllm-qwen3-8b.yaml"
  else
    local server
    server="$(sed -e "s/fin-agent-vllm-qwen3-8b/fin-agent-vllm-${name}/g" \
      -e "s|--max-model-len 40960|--max-model-len 40960 ${VLLM_EXTRA_ARGS}|" \
      "${SCRIPT_DIR}/vllm-qwen3-8b.yaml")"
    case "${server}" in
      *"--quantization ${QUANTIZATION}"*) ;;
      *) echo "ERROR: failed to inject '${VLLM_EXTRA_ARGS}' into vllm-qwen3-8b.yaml -- its --max-model-len line must have changed" >&2; exit 1 ;;
    esac
    echo "${server}" | kubectl apply -f -
  fi
  kubectl -n "${NAMESPACE}" rollout status "deploy/fin-agent-vllm-${name}" --timeout=900s
  sed -e "s/__TEST_CATEGORIES__/${TEST_CATEGORIES}/g" -e "s/__VARIANT__/${VARIANT}/g" \
    "${SCRIPT_DIR}/bfcl-eval-baseline-categories-job.yaml" | kubectl apply -f -
  kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/fin-agent-bfcl-eval-${name}" --timeout="${WAIT_TIMEOUT}"
  kubectl -n "${NAMESPACE}" logs -l "app=fin-agent-bfcl-eval-${name}" --tail=20
  kubectl -n "${NAMESPACE}" delete deployment "fin-agent-vllm-${name}" --ignore-not-found
  kubectl -n "${NAMESPACE}" delete service "fin-agent-vllm-${name}" --ignore-not-found
}

run_mlflow_checkpoint() {
  # VARIANT rides on __NAME__, which already drives every identifier in this manifest
  # (Deployment, Service, Job, --bfcl-project-root, --mlflow-experiment-name), so a
  # quantized run isolates itself everywhere for free.
  local name="$1${VARIANT}" run_id="$2"
  echo ">> [${name}] MLflow run_id=${run_id}"
  local rendered
  rendered="$(sed -e "s/__NAME__/${name}/g" -e "s/__RUN_ID__/${run_id}/g" \
    -e "s/__TEST_CATEGORIES__/${TEST_CATEGORIES}/g" \
    -e "s|__VLLM_EXTRA_ARGS__|${VLLM_EXTRA_ARGS}|g" \
    "${SCRIPT_DIR}/bfcl-eval-mlflow-checkpoint-job.yaml")"
  kubectl -n "${NAMESPACE}" delete job "fin-agent-bfcl-eval-${name}" --ignore-not-found --wait=true
  kubectl -n "${NAMESPACE}" delete deployment "fin-agent-vllm-${name}" --ignore-not-found --wait=true
  echo "${rendered}" | kubectl apply -f -
  kubectl -n "${NAMESPACE}" rollout status "deploy/fin-agent-vllm-${name}" --timeout=900s
  kubectl -n "${NAMESPACE}" wait --for=condition=complete "job/fin-agent-bfcl-eval-${name}" --timeout="${WAIT_TIMEOUT}"
  kubectl -n "${NAMESPACE}" logs -l "app=fin-agent-bfcl-eval-${name}" --tail=20
  # Deployment/Service only -- NOT the whole rendered manifest, which also contains the
  # fin-agent-bfcl-mlflow-results PVC. `kubectl delete -f -` on the full thing deletes
  # results/scores generated this run along with the server, same mistake
  # bfcl-eval-job.yaml's own header explicitly calls out avoiding ("The PVC is
  # deliberately not deleted here so a finished run's results survive"). Losing them
  # here is worse than usual: a bug fix that only affects the *logging* step (e.g. the
  # read_category_summaries model-path bug) would otherwise force a full, expensive
  # regeneration to recover, when a quick re-score of already-generated results would
  # have done it in about a minute.
  kubectl -n "${NAMESPACE}" delete deployment "fin-agent-vllm-${name}" --ignore-not-found
  kubectl -n "${NAMESPACE}" delete service "fin-agent-vllm-${name}" --ignore-not-found
}

for model in ${MODELS}; do
  case "${model}" in
    baseline) run_baseline ;;
    sft)      run_mlflow_checkpoint sft "${SFT_RUN_ID}" ;;
    grpo)     run_mlflow_checkpoint grpo "${GRPO_RUN_ID}" ;;
    *)        echo "unknown model '${model}' in MODELS -- expected: baseline sft grpo" >&2; exit 1 ;;
  esac
done

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
