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
import sys

import mlflow


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

    if os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"[serve_mlflow_checkpoint] {model_dir} already downloaded, skipping fetch", file=sys.stderr)
    else:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        print(f"[serve_mlflow_checkpoint] downloading run {args.run_id}'s '{args.artifact_path}' artifact ...", file=sys.stderr)
        downloaded = mlflow.artifacts.download_artifacts(
            run_id=args.run_id, artifact_path=args.artifact_path, dst_path=dst_dir
        )
        # download_artifacts nests its output under a directory named after
        # artifact_path -- verified directly against mlflow, not assumed (same finding
        # documented in vllm-grpo-mtp.yaml's initContainer): dst_path=X,
        # artifact_path="model" returns X/model, i.e. exactly model_dir above. This
        # assert exists to catch mlflow ever changing that behavior loudly, rather than
        # silently pointing vllm at the wrong directory.
        assert downloaded == model_dir, (
            f"expected mlflow to place the download at {model_dir}, got {downloaded} -- "
            "mlflow.artifacts.download_artifacts' nesting behavior may have changed."
        )

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
