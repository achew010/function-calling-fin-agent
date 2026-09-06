#!/usr/bin/env bash
# Runs the BFCL eval end-to-end: applies the vLLM inference server it depends on
# (./vllm-qwen3-8b.yaml), waits for it to be Ready, then applies and follows
# the eval Job. bfcl-eval-job.yaml itself does NOT bring up the server it needs — it
# only polls for one that's already there — so this script exists precisely to make sure
# that dependency isn't forgotten when running the benchmark on its own.
#
# Usage:
#   ./run-bfcl-eval.sh
#
# The vLLM server is left running afterward (it's a Deployment, not a one-off) — tear it
# down with:
#   kubectl -n fin-agent delete -f ./vllm-qwen3-8b.yaml

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE=fin-agent

echo ">> Applying the vLLM inference server (configs/templates/inference/vllm-qwen3-8b.yaml)"
kubectl apply -f "${SCRIPT_DIR}/vllm-qwen3-8b.yaml"

echo ">> Waiting for it to become Ready (model download + load can take several minutes)..."
kubectl -n "${NAMESPACE}" rollout status deploy/fin-agent-vllm-qwen3-8b --timeout=900s

echo ">> Applying the BFCL eval Job"
# Job selectors are immutable, so a prior Job of the same name must go before reapplying
# — this is safe to do on every run now: results/scores/MLflow-run-id live on the
# fin-agent-bfcl-results PVC (not deleted here), so a rerun resumes from where a killed
# or completed prior Job left off rather than starting over.
kubectl -n "${NAMESPACE}" delete job fin-agent-bfcl-eval --ignore-not-found
kubectl apply -f "${SCRIPT_DIR}/bfcl-eval-job.yaml"

echo ">> Following logs (Ctrl-C stops watching, the job keeps running)..."
kubectl -n "${NAMESPACE}" wait --for=condition=ready pod -l app=fin-agent-bfcl-eval --timeout=120s || true
kubectl -n "${NAMESPACE}" logs -l app=fin-agent-bfcl-eval -f
