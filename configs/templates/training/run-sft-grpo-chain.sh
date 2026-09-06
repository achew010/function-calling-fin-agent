#!/usr/bin/env bash
# Runs the real SFT -> real GRPO chain end-to-end: prepares the dataset, runs the real
# SFT training run, then the real GRPO run against its checkpoint -- exercising the exact
# handoff grpo-job.yaml depends on (--base-model /checkpoints/sft, mounted read-write
# from sft-job.yaml's hostPath output) in one command. See run-grpo-smoke-chain.sh for
# the same idea applied to the two smoke tests -- run that first to prove the pipeline
# and the checkpoint handoff both work end-to-end before committing GPU hours to this one.
#
# Usage:
#   ./run-sft-grpo-chain.sh
#
# This is genuinely long-running: sft-job.yaml's own activeDeadlineSeconds is 24h,
# grpo-job.yaml's is a generous, unvalidated 48h (see its header for why) -- the waits
# below match those ceilings, so this script can legitimately block for days. Run it in
# a way that survives your terminal disconnecting (screen/tmux/nohup), not in a shell
# you're staying attached to throughout.
#
# Each stage's Job is deleted and reapplied if a prior one exists: Job pod templates are
# immutable, so a bare `kubectl apply` on an already-existing Job silently keeps using
# its OLD template instead of erroring or updating -- see run-grpo-smoke-chain.sh's
# header for the real debugging time this exact trap already cost once in this project.
# prepare-dataset-job.yaml is deterministic/seeded, so rerunning it here every time is
# safe and just confirms the data is present rather than skipping a real check.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE=fin-agent

echo ">> [1/3] Preparing the dataset"
kubectl -n "${NAMESPACE}" delete job fin-agent-prepare-dataset --ignore-not-found --wait=true
kubectl apply -f "${SCRIPT_DIR}/prepare-dataset-job.yaml"
kubectl -n "${NAMESPACE}" wait --for=condition=complete job/fin-agent-prepare-dataset --timeout=1800s

echo ">> [2/3] Running the real SFT training run (this can take hours)"
kubectl -n "${NAMESPACE}" delete job fin-agent-sft --ignore-not-found --wait=true
kubectl apply -f "${SCRIPT_DIR}/sft-job.yaml"
kubectl -n "${NAMESPACE}" wait --for=condition=complete job/fin-agent-sft --timeout=86400s

echo ">> [3/3] Running the real GRPO run against that checkpoint (this can take much longer)"
kubectl -n "${NAMESPACE}" delete job fin-agent-grpo --ignore-not-found --wait=true
kubectl apply -f "${SCRIPT_DIR}/grpo-job.yaml"

echo ">> Following GRPO logs (Ctrl-C stops watching, the job keeps running)..."
kubectl -n "${NAMESPACE}" wait --for=condition=ready pod -l app=fin-agent-grpo --timeout=120s || true
kubectl -n "${NAMESPACE}" logs -l app=fin-agent-grpo -f
