"""Serve a specific MLflow run's model artifact via vLLM -- the no-kube analogue of
configs/templates/inference/vllm-grpo-mtp.yaml's initContainer (download by run_id) +
vllm serve, collapsed into one process since there's no separate init/main container
split to put them in on a bare host.

Usage:
    python serve_mlflow_checkpoint.py --run-id e783b52f2b7a42cc8ff2b2786949cb71 \
        --mlflow-tracking-uri http://mlflow.fin-agent.svc.cluster.local:5000

Any extra arguments (e.g. --dtype, --tensor-parallel-size) are passed straight through
to `vllm serve` unmodified.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import mlflow

# configs/setup/mlflow.yaml deliberately configures this project's MLflow with a bare
# local-path artifact root (--default-artifact-root /data/artifacts), not the
# HTTP-proxied mlflow-artifacts:/ scheme -- see that file's own header for why
# (sidesteps a proxy-scheme resolution bug by having every pod mount the identical PVC
# at the identical in-container path). That means every run's artifact_uri looks like
# /data/artifacts/<experiment_id>/<run_id>/artifacts -- resolvable by
# mlflow.artifacts.download_artifacts() only from *inside* that same mount (any pod),
# never from a bare host, which has no /data/artifacts directory at all (confirmed for
# real: MlflowException "Failed to download artifacts from path 'model', please ensure
# that the path is correct" running this from outside the cluster). The same
# setup/mlflow.yaml PersistentVolume also names where that PVC's data really lives on
# the node's own disk -- so when the normal client-side download fails this specific
# way, read directly from there instead of going through mlflow's (here, unusable)
# artifact-repository machinery.
INCLUSTER_ARTIFACT_ROOT = "/data/artifacts"
HOST_ARTIFACT_ROOT = "/var/lib/fin-agent/mlflow-artifacts"


def _hostpath_fallback(run_id: str, artifact_path: str) -> str | None:
    """The real, on-disk directory for one run's artifact, if this project's known
    in-cluster-path -> hostPath mapping applies -- None if the run's own artifact_uri
    doesn't match that convention at all (a different/misconfigured MLflow instance),
    so callers don't misapply a fix that's specific to this project's setup.
    """
    run = mlflow.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    if not artifact_uri.startswith(INCLUSTER_ARTIFACT_ROOT):
        return None
    host_run_dir = HOST_ARTIFACT_ROOT + artifact_uri[len(INCLUSTER_ARTIFACT_ROOT) :]
    return os.path.join(host_run_dir, artifact_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="MLflow run id to pull the model artifact from.")
    parser.add_argument("--mlflow-tracking-uri", required=True)
    parser.add_argument(
        "--artifact-path",
        default="model",
        help="Artifact subpath within the run. Matches train_sft.py/train_grpo.py's own "
        "mlflow.log_artifacts(..., artifact_path='model') -- change only if a run logged "
        "its checkpoint somewhere else.",
    )
    parser.add_argument(
        "--dst-dir",
        default=None,
        help="Where to download the checkpoint. Defaults to "
        "~/.cache/fin-agent/mlflow-models/<run_id>, and skips the download entirely if "
        "that already has a config.json -- so pointing at the same run_id twice doesn't "
        "re-pull a multi-GB checkpoint.",
    )
    parser.add_argument(
        "--served-model-name",
        default="Qwen/Qwen3-8B",
        help="Not the local path -- matches run_bfcl_eval.py's MODEL_CONFIG_MAPPING "
        "lookup (keyed on model id string, not on the weights actually being served), "
        "same reasoning as bfcl-eval-mlflow-checkpoint-job.yaml.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=40960,
        help="Qwen3-8B's real native context -- see vllm-qwen3-8b.yaml's header for why "
        "a lower cap here silently breaks BFCL scoring (it sizes requests off the "
        "model's own HF config, not off whatever this server was actually given).",
    )
    args, extra_vllm_args = parser.parse_known_args()

    dst_dir = args.dst_dir or os.path.expanduser(f"~/.cache/fin-agent/mlflow-models/{args.run_id}")
    model_dir = os.path.join(dst_dir, args.artifact_path)
    # Written only after a fully successful download/copy -- NOT the same check as
    # "does config.json exist", which a partial copy (e.g. one file mid-copytree
    # hitting a permission error, see the fallback branch below) can satisfy while
    # still missing the actual weight files. Confirmed for real: a crashed copytree
    # left config.json in place but not model.safetensors, and a config.json-only
    # check on the next run wrongly called that "already downloaded" and sent vLLM at
    # an incomplete directory ("Cannot find any model weights").
    sentinel = os.path.join(model_dir, ".fin_agent_download_complete")

    if os.path.exists(sentinel):
        print(f"[serve_mlflow_checkpoint] {model_dir} already downloaded, skipping fetch", file=sys.stderr)
    else:
        # Clear any stale partial state before retrying, whichever path below actually
        # runs -- copytree (and download_artifacts) don't guarantee a clean directory
        # to write into, and we specifically got here because a previous attempt may
        # have left one half-populated.
        shutil.rmtree(model_dir, ignore_errors=True)
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        print(f"[serve_mlflow_checkpoint] downloading run {args.run_id}'s '{args.artifact_path}' artifact ...", file=sys.stderr)
        try:
            downloaded = mlflow.artifacts.download_artifacts(
                run_id=args.run_id, artifact_path=args.artifact_path, dst_path=dst_dir
            )
            # download_artifacts nests its output under a directory named after
            # artifact_path -- verified directly against mlflow, not assumed (same
            # finding documented in vllm-grpo-mtp.yaml's initContainer): dst_path=X,
            # artifact_path="model" returns X/model, i.e. exactly model_dir above.
            # This assert exists to catch mlflow ever changing that behavior loudly,
            # rather than silently pointing vllm at the wrong directory.
            assert downloaded == model_dir, (
                f"expected mlflow to place the download at {model_dir}, got {downloaded} -- "
                "mlflow.artifacts.download_artifacts' nesting behavior may have changed."
            )
        except Exception as e:
            host_path = _hostpath_fallback(args.run_id, args.artifact_path)
            if host_path is None or not os.path.isdir(host_path):
                raise
            print(
                f"[serve_mlflow_checkpoint] normal download failed ({e}); this MLflow "
                f"instance uses a bare local-path artifact store (see this file's "
                f"INCLUSTER_ARTIFACT_ROOT comment) -- falling back to reading directly "
                f"from {host_path}",
                file=sys.stderr,
            )
            shutil.copytree(host_path, model_dir)
        # Only reached if the try block (or its except fallback) completed without
        # raising -- an interrupted copy leaves no sentinel, so the next run correctly
        # treats it as incomplete and retries from a clean directory instead of
        # trusting a partial one.
        open(sentinel, "w").close()

    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise SystemExit(
            f"no config.json in {model_dir} -- did run {args.run_id} actually log a "
            f"'{args.artifact_path}' artifact (train_sft.py/train_grpo.py with --mlflow)?"
        )

    cmd = [
        "vllm",
        "serve",
        model_dir,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        *extra_vllm_args,
    ]
    print(f"[serve_mlflow_checkpoint] $ {' '.join(cmd)}", file=sys.stderr)
    os.execvp("vllm", cmd)


if __name__ == "__main__":
    main()
