#!/usr/bin/env bash
# Runs the full vLLM latency benchmark (concurrencies 1/8/16/32/64, --num-prompts
# NUM_PROMPTS) against four serving configurations of the SAME checkpoint, one at a time:
#   1. bf16                        -- unquantized baseline
#   2. fp8_per_tensor              -- FP8 online-quantized weights
#   3. fp8_weights_and_kv          -- FP8 weights + FP8 KV-cache
#   4. fp8_weights_kv_dflash       -- FP8 weights + FP8 KV-cache + DFlash speculative
#                                      decoding (z-lab/Qwen3-8B-DFlash-b16 drafter, 4
#                                      candidate tokens) -- NOT MTP: MTP needs a trained
#                                      MTP head on the target checkpoint, which this
#                                      project's fine-tunes don't have (see
#                                      ./vllm-serve-checkpoint-fp8-dflash.yaml's header)
#
# Each variant becomes its own MLflow run in the fin-agent-vllm-bench experiment (see
# 2_evaluations/log_vllm_bench_to_mlflow.py), labelled by the names above via
# --label -- compare them directly by that label. Also logs kv_cache_usage_perc_mean/_max
# per concurrency (scraped from vLLM's own /metrics), so variants 2-4 are checkable
# against the real cache-headroom effect, not just throughput/latency.
#
# Single GPU, one workload at a time (this project's standing constraint, see root
# README's "Design notes"): each Deployment is torn down before the next is applied, and
# an EXIT trap tears the last one down too, deliberate Ctrl-C included.
#
# This runs `tox -e vllm-bench` from THIS bare host (not in-cluster -- see that env's own
# tox.ini comment for why), so it needs a local port-forward to whichever Deployment is
# currently up; this script manages that itself. Run
# `kubectl -n fin-agent port-forward svc/mlflow 5000:5000` in a separate terminal first
# and leave it running -- this script does not manage that one.
#
# Usage:
#   SFT_RUN_ID=<mlflow run id of the SFT checkpoint to benchmark> \
#     ./run-vllm-bench-quantization-suite.sh
#   NUM_PROMPTS=1000 CONCURRENCIES=1,8,16,32,64 SFT_RUN_ID=<run id> \
#     ./run-vllm-bench-quantization-suite.sh   # same as the defaults, spelled out
#
# Prerequisites: setup/setup.sh, and the MLflow run at SFT_RUN_ID must have a "model"
# artifact (train_sft.py/train_grpo.py with --mlflow both log one) -- see
# ./vllm-serve-checkpoint.yaml's own header for the fuller explanation.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE=fin-agent
SFT_RUN_ID="${SFT_RUN_ID:?set SFT_RUN_ID=<mlflow run id of the checkpoint to benchmark>}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
CONCURRENCIES="${CONCURRENCIES:-1,8,16,32,64}"
BFCL_CATEGORIES="${BFCL_CATEGORIES:-simple,multiple,parallel,parallel_multiple}"

# Every Deployment/Service name this script ever creates -- used to tear down whichever
# one is currently up before applying the next, and at exit. Job pod templates are
# immutable elsewhere in this project, but these are Deployments (mutable in principle) --
# still torn down explicitly rather than reconfigured in place, matching every other
# variant-switch in this repo (see root README's step 5).
ALL_NAMES=(sft sft-fp8 sft-fp8-kvfp8 sft-fp8-dflash)

teardown_all() {
  local deployments=() services=()
  for n in "${ALL_NAMES[@]}"; do
    deployments+=("fin-agent-vllm-${n}")
    services+=("fin-agent-vllm-${n}")
  done
  kubectl -n "${NAMESPACE}" delete deployment "${deployments[@]}" --ignore-not-found --wait=true
  kubectl -n "${NAMESPACE}" delete service "${services[@]}" --ignore-not-found
  pkill -f "port-forward svc/fin-agent-vllm" || true
}
trap teardown_all EXIT

wait_healthy() {
  local name="$1"
  kubectl -n "${NAMESPACE}" port-forward "svc/fin-agent-vllm-${name}" 8000:8000 &
  for i in $(seq 1 30); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "fin-agent-vllm-${name} never became healthy on :8000" >&2
  exit 1
}

run_variant() {
  local name="$1" template="$2" extra_args="$3" label="$4"
  echo ">> [${label}] tearing down previous variant, applying fin-agent-vllm-${name} ..."
  teardown_all
  sed -e "s/__NAME__/${name}/g" -e "s/__RUN_ID__/${SFT_RUN_ID}/g" -e "s/__VLLM_EXTRA_ARGS__/${extra_args}/g" \
    "${SCRIPT_DIR}/${template}" | kubectl apply -f -
  kubectl -n "${NAMESPACE}" rollout status "deploy/fin-agent-vllm-${name}" --timeout=900s
  wait_healthy "${name}"

  echo ">> [${label}] benchmarking (num-prompts=${NUM_PROMPTS}, concurrencies=${CONCURRENCIES}) ..."
  tox -e vllm-bench -- \
    --base-url http://localhost:8000 --model Qwen/Qwen3-8B \
    --dataset-name hf --dataset-path gorilla-llm/Berkeley-Function-Calling-Leaderboard \
    --bfcl-categories "${BFCL_CATEGORIES}" \
    --num-prompts "${NUM_PROMPTS}" --concurrencies "${CONCURRENCIES}" --label "${label}"
}

echo ">> Make sure 'kubectl -n ${NAMESPACE} port-forward svc/mlflow 5000:5000' is already running in another terminal."

# 1. Baseline
run_variant "sft" "vllm-serve-checkpoint.yaml" "" "bf16"

# 2. FP8 online-quantized weights
run_variant "sft-fp8" "vllm-serve-checkpoint.yaml" "--quantization fp8_per_tensor" "fp8_per_tensor"

# 3. FP8 weights + FP8 KV-cache
run_variant "sft-fp8-kvfp8" "vllm-serve-checkpoint.yaml" \
  "--quantization fp8_per_tensor --kv-cache-dtype fp8_e4m3" "fp8_weights_and_kv"

# 4. FP8 weights + FP8 KV-cache + DFlash speculative decoding -- fully baked into its own
# template (quantization, kv-cache dtype, --trust-remote-code, --speculative-config all
# hardcoded), not routed through __VLLM_EXTRA_ARGS__ -- see that file's header for why
# (the drafter's model id has a literal "/" that collides with sed's own delimiter).
# extra_args is empty and unused here; the template has no __VLLM_EXTRA_ARGS__ to fill.
run_variant "sft-fp8-dflash" "vllm-serve-checkpoint-fp8-dflash.yaml" "" "fp8_weights_kv_dflash"

echo ""
echo "=============================================================="
echo " Suite complete. Compare labels bf16 / fp8_per_tensor /"
echo " fp8_weights_and_kv / fp8_weights_kv_dflash in MLflow, experiment"
echo " fin-agent-vllm-bench -- each carries kv_cache_usage_perc_mean/_max"
echo " alongside throughput/TTFT/TPOT for every concurrency point."
echo "=============================================================="
