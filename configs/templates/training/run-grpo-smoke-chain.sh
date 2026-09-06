#!/usr/bin/env bash
# Runs the SFT-smoke -> GRPO-smoke chain end-to-end: prepares the dataset, runs the SFT
# smoke test, then the GRPO smoke test against its checkpoint -- exercising the exact
# handoff grpo-job.yaml depends on (--base-model /checkpoints/smoke-test, mounted
# read-only from smoke-test-job.yaml's hostPath output) in one command, rather than
# running each stage by hand and having to remember the order and wait between them.
#
# Usage:
#   ./run-grpo-smoke-chain.sh
#
# Each stage's Job is deleted and reapplied if a prior one exists: Job pod templates are
# immutable, so a bare `kubectl apply` on an already-existing Job silently keeps using
# its OLD template instead of erroring or updating -- this is the exact bug that cost
# real debugging time earlier in this project (a stale smoke-test-job.yaml kept running
# pre-merge-fix code and an unmounted MLflow artifacts volume for several reruns before
# it was caught). prepare-dataset-job.yaml is deterministic/seeded, so rerunning it here
# every time is safe and just confirms the data is present rather than skipping a real
# check.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE=fin-agent

echo ">> [1/3] Preparing the dataset"
kubectl -n "${NAMESPACE}" delete job fin-agent-prepare-dataset --ignore-not-found --wait=true
kubectl apply -f "${SCRIPT_DIR}/prepare-dataset-job.yaml"
kubectl -n "${NAMESPACE}" wait --for=condition=complete job/fin-agent-prepare-dataset --timeout=1800s

echo ">> [2/3] Running the SFT smoke test"
kubectl -n "${NAMESPACE}" delete job fin-agent-sft-smoke-test --ignore-not-found --wait=true
kubectl apply -f "${SCRIPT_DIR}/smoke-test-job.yaml"
kubectl -n "${NAMESPACE}" wait --for=condition=complete job/fin-agent-sft-smoke-test --timeout=1800s

echo ">> [3/3] Running the GRPO smoke test against that checkpoint"
kubectl -n "${NAMESPACE}" delete job fin-agent-grpo-smoke-test --ignore-not-found --wait=true
kubectl apply -f "${SCRIPT_DIR}/grpo-job.yaml"

echo ">> Following GRPO logs (Ctrl-C stops watching, the job keeps running)..."
kubectl -n "${NAMESPACE}" wait --for=condition=ready pod -l app=fin-agent-grpo-smoke-test --timeout=120s || true
kubectl -n "${NAMESPACE}" logs -l app=fin-agent-grpo-smoke-test -f
