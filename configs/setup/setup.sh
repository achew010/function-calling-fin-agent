#!/usr/bin/env bash
# configs/setup.sh — bootstrap a barebones Linux VM (single GPU, e.g. 1xH100) into a
# single-node k3s cluster capable of running this project's Kubernetes jobs
# (smoke-test-job.yaml, bfcl-eval.yaml, grpo-job.yaml). Run ON that VM.
#
# What this does:
#   1. If an NVIDIA GPU is present (`nvidia-smi` works): installs nvidia-container-toolkit
#      *before* k3s, since k3s auto-detects it at install/start time and registers a
#      "nvidia" containerd runtime on its own — no manual containerd config editing needed.
#   2. Installs k3s (single-node server; workloads run on this same node) if not already
#      present, and points kubectl at it.
#   3. Applies the nvidia RuntimeClass + device plugin, so Kubernetes can schedule
#      nvidia.com/gpu resource requests. No time-slicing: this project runs one workload
#      at a time on the one GPU (a training job, or a vLLM inference job — never both),
#      so one slot for the one physical GPU is the right default (see
#      nvidia-device-plugin.yaml if you want to change that).
#   4. Creates the fin-agent namespace and deploys a self-contained MLflow
#      instance for the jobs to log to.
#
# Usage:
#   ./setup.sh
#
# Verify afterward:
#   kubectl get nodes
#   kubectl describe node $(hostname) | grep nvidia.com/gpu   # only if a GPU is present
#   kubectl -n fin-agent get pods

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

HAS_GPU=false
if command -v nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=true
  echo ">> NVIDIA GPU detected:"
  nvidia-smi -L
  if ! command -v nvidia-container-runtime >/dev/null 2>&1 && \
     ! command -v nvidia-container-toolkit >/dev/null 2>&1; then
    echo ">> Installing nvidia-container-toolkit (before k3s, so k3s auto-detects it)"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
      sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
  else
    echo ">> nvidia-container-toolkit already installed, skipping"
  fi
else
  echo ">> No NVIDIA GPU detected (nvidia-smi not found)."
  echo "   Every job manifest here requests nvidia.com/gpu and won't schedule without a"
  echo "   GPU node — install the NVIDIA driver first if this VM does have one."
fi

if ! command -v k3s >/dev/null 2>&1; then
  echo ">> Installing k3s (single-node server)"
  curl -sfL https://get.k3s.io | sh -s - server --write-kubeconfig-mode 644
  mkdir -p "$HOME/.kube"
  sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
  sudo chown "$(id -u):$(id -g)" "$HOME/.kube/config"
  export KUBECONFIG="$HOME/.kube/config"
else
  echo ">> k3s already installed — restarting so it re-detects the NVIDIA runtime if the toolkit was just installed"
  sudo systemctl restart k3s
fi

echo ">> Waiting for the node to become Ready..."
until kubectl wait --for=condition=Ready node --all --timeout=10s >/dev/null 2>&1; do
  sleep 3
done

if [ "$HAS_GPU" = true ]; then
  if sudo grep -q 'nvidia' /var/lib/rancher/k3s/agent/etc/containerd/config.toml 2>/dev/null; then
    echo ">> NVIDIA runtime registered with containerd."
  else
    echo "WARNING: NVIDIA runtime not found in containerd config yet — the device plugin"
    echo "         may CrashLoop until this resolves. Try: sudo systemctl restart k3s"
  fi

  echo ">> Applying the nvidia RuntimeClass + device plugin"
  kubectl apply -f "${SCRIPT_DIR}/nvidia-runtimeclass.yaml"
  kubectl apply -f "${SCRIPT_DIR}/nvidia-device-plugin.yaml"

  echo ">> Waiting for nvidia.com/gpu to appear as an allocatable resource..."
  for i in $(seq 1 30); do
    if kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' 2>/dev/null | grep -qv '^ *$'; then
      echo ">> GPU resource visible to Kubernetes."
      break
    fi
    sleep 5
  done
fi

echo ">> Creating namespace and deploying MLflow"
kubectl apply -f "${SCRIPT_DIR}/namespace.yaml"
kubectl apply -f "${SCRIPT_DIR}/mlflow.yaml"
kubectl -n fin-agent rollout status deploy/mlflow --timeout=180s
kubectl config set-context --current --namespace=fin-agent

# Every job manifest under templates/ mounts its code from a ConfigMap rather than a
# custom image (see configs/README.md's design notes) — built here, once, from
# PROJECT_ROOT, so `kubectl apply -f templates/...` just works afterward instead of
# failing with FailedMount/ContainerCreating until someone remembers to build it by hand
# (the exact failure mode that motivated adding this). rebuild-configmaps.sh is
# idempotent (--dry-run=client -o yaml | kubectl apply -f - per ConfigMap), so rerunning
# it here (or standalone, after any later `git pull`) safely updates each one in place —
# see that script's own header for why re-running it is not optional after a pull.
echo ">> Building job source ConfigMaps"
"${SCRIPT_DIR}/rebuild-configmaps.sh"

echo ""
echo "=============================================================="
echo " Cluster ready. Job source ConfigMaps built:"
echo "   fin-agent-data-prep-src, fin-agent-sft-src, fin-agent-grpo-src, fin-agent-bfcl-src"
echo " MLflow (in-cluster, used by the job manifests):"
echo "   http://mlflow.fin-agent.svc.cluster.local:5000"
echo " MLflow (from this machine, for the UI):"
echo "   kubectl -n fin-agent port-forward svc/mlflow 5000:5000"
echo "   then open http://localhost:5000"
echo ""
echo " Next: prepare the dataset once (everything else reads from it), then apply"
echo " whichever job you want — see README.md:"
echo "   kubectl apply -f ${SCRIPT_DIR}/../templates/training/prepare-dataset-job.yaml"
echo "   kubectl wait --for=condition=complete job/fin-agent-prepare-dataset -n fin-agent --timeout=1800s"
echo "=============================================================="
