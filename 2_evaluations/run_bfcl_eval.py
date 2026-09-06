"""BFCL (Berkeley Function-Calling Leaderboard) evaluator.

Follows this project's Evaluator interface: load_eval_dataset -> run_predictions ->
compute_metrics, implemented on top of the real `bfcl-eval` CLI/package (command `bfcl`).
By default this lets BFCL self-host the model: `bfcl generate` launches its own
vLLM/SGLang server, downloads weights from the HF Hub if needed, generates, and tears
the server down — no separate serving step required. Pass skip_server_setup=True plus
vllm_endpoint/vllm_port to instead point at an already-running OpenAI-compatible server.

Default model: Qwen/Qwen3-8B — confirmed via `bfcl models` / MODEL_CONFIG_MAPPING as the
"Prompt"-style registration (is_fc_model=False), which matches this project's own
prompting-based fine-tuning approach (tools embedded as text in the system message,
calls parsed out of generated text) — NOT the "-FC" variant, which drives the model's
native tool-calling API format instead. This should be the SAME base checkpoint
1_training/1_sft/train_sft.py fine-tunes, so a baseline-vs-fine-tuned comparison isolates
the effect of fine-tuning rather than confounding it with a different base model.

To evaluate a fine-tuned checkpoint: pass local_model_path pointing at train_sft.py's
--output-dir directly, keeping model="Qwen/Qwen3-8B" (same architecture/tokenizer/
handler). No separate merge step needed — train_sft.py already saves a merged, standalone
model (not a bare LoRA adapter), loadable the same way as the raw HF baseline.

Usage:
    python run_bfcl_eval.py --model Qwen/Qwen3-8B --test-category python --num-gpus 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import warnings
from abc import ABC, abstractmethod
from typing import Any

from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING, TEST_FILE_MAPPING
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING


class Evaluator(ABC):
    """load_eval_dataset -> run_predictions -> compute_metrics pipeline for scoring a
    model against a benchmark."""

    @abstractmethod
    def load_eval_dataset(self) -> Any:
        """Resolve/validate what the evaluation will run against."""

    @abstractmethod
    def run_predictions(self, eval_dataset: Any) -> Any:
        """Run the model against the eval dataset, producing raw predictions."""

    @abstractmethod
    def compute_metrics(self, predictions: Any) -> Any:
        """Score the predictions and report metrics."""


class BFCLEvaluator(Evaluator):
    """Evaluator for BFCL, via the real `bfcl-eval` CLI.

    `model` accepts either a Hugging Face model id BFCL knows how to self-host (e.g.
    "Qwen/Qwen3-8B") or, with `local_model_path` set, a local checkpoint directory (e.g.
    a merged fine-tuned model) served under that same model id/architecture — see module
    docstring for why the id still has to be a registered one either way (BFCL picks its
    prompt-formatting/response-parsing handler off the id, not off the weights).
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-8B",
        test_category: str = "python",
        local_model_path: str | None = None,
        num_gpus: int = 1,
        num_threads: int = 1,
        gpu_memory_utilization: float = 0.9,
        backend: str = "vllm",
        skip_server_setup: bool = False,
        vllm_endpoint: str = "localhost",
        vllm_port: str = "8000",
        result_dir: str | None = None,
        score_dir: str | None = None,
    ) -> None:
        if model not in MODEL_CONFIG_MAPPING:
            raise ValueError(f"Unknown BFCL model id {model!r} — run `bfcl models` to see valid ids.")
        if MODEL_CONFIG_MAPPING[model].is_fc_model:
            warnings.warn(
                f"{model!r} is a '-FC' (native function-calling API) registration. "
                "This project's fine-tune uses prompting-style tool calls (tools as "
                "text, calls parsed from generated text) — the non-'-FC' id is almost "
                "certainly what you want for an apples-to-apples comparison."
            )
        self.model = model
        self.test_category = test_category
        self.local_model_path = local_model_path
        self.num_gpus = num_gpus
        self.num_threads = num_threads
        self.gpu_memory_utilization = gpu_memory_utilization
        self.backend = backend
        self.skip_server_setup = skip_server_setup
        self.vllm_endpoint = vllm_endpoint
        self.vllm_port = vllm_port
        self.result_dir = result_dir
        self.score_dir = score_dir

    def load_eval_dataset(self) -> dict[str, str]:
        """Resolve test_category into BFCL's own concrete test names and backing files.

        BFCL owns its eval data internally — this project doesn't supply it — so this
        method's job is making "what will actually run" explicit rather than opaque,
        and failing fast on a typo'd category before spending GPU time on run_predictions.
        """
        if self.test_category in TEST_COLLECTION_MAPPING:
            test_names = TEST_COLLECTION_MAPPING[self.test_category]
        elif self.test_category in TEST_FILE_MAPPING:
            test_names = [self.test_category]
        else:
            raise ValueError(
                f"Unknown test_category {self.test_category!r} — run `bfcl "
                "test-categories` to see valid groups/names."
            )
        return {name: TEST_FILE_MAPPING[name] for name in test_names}

    def run_predictions(self, eval_dataset: dict[str, str]) -> None:
        """`bfcl generate` — runs self.model against every test file in eval_dataset."""
        cmd = [
            "bfcl",
            "generate",
            "--model",
            self.model,
            "--test-category",
            self.test_category,
            "--num-gpus",
            str(self.num_gpus),
            "--num-threads",
            str(self.num_threads),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--backend",
            self.backend,
        ]
        if self.local_model_path:
            cmd += ["--local-model-path", self.local_model_path]
        if self.result_dir:
            cmd += ["--result-dir", self.result_dir]

        env = os.environ.copy()
        if self.skip_server_setup:
            cmd.append("--skip-server-setup")
            env["VLLM_ENDPOINT"] = self.vllm_endpoint
            env["VLLM_PORT"] = self.vllm_port

        self._run(cmd, env=env)

    def compute_metrics(self, predictions: None = None) -> None:
        """`bfcl evaluate` scores the generated results and writes one
        BFCL_v3_<category>_score.json per category — everything log_bfcl_to_mlflow.py's
        read_category_summaries() needs. `predictions` is unused — bfcl reads
        run_predictions' output back from disk (--result-dir) itself rather than taking
        it in-process.

        Deliberately does NOT call `bfcl scores`: that subcommand builds a fixed-column
        leaderboard table (data_non_live.csv etc.) that unconditionally expects an
        "exec_*" category column ("Non-Live Exec Acc") to exist — `bfcl
        __main__.py:scores`, `headers.index(col) for col in selected_columns` with no
        guard. This project's test_category ("python") only ever runs the
        non-exec/non-multi-turn categories (see category_mapping.py — the exec_* entries
        are commented out), so that column never exists and `bfcl scores` crashes with
        `ValueError: 'Non-Live Exec Acc' is not in list` every time, regardless of how
        much of the suite finished. Confirmed for real running this against a live
        result set. Nothing here reads its CSV output anyway.
        """
        evaluate_cmd = ["bfcl", "evaluate", "--model", self.model, "--test-category", self.test_category]
        if self.result_dir:
            evaluate_cmd += ["--result-dir", self.result_dir]
        if self.score_dir:
            evaluate_cmd += ["--score-dir", self.score_dir]
        self._run(evaluate_cmd)

    @staticmethod
    def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-8B",
        help="BFCL model id — `bfcl models` lists all supported ids. Use the "
        "non-'-FC' variant to match this project's prompting-based call format.",
    )
    parser.add_argument(
        "--test-category",
        default="python",
        help="`bfcl test-categories` lists groups; 'python' is BFCL's own "
        "Python-relevant subset.",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Concurrent request threads BFCL's own harness uses — bfcl-eval's default "
        "of 1 sends requests serially even when the server (vLLM's continuous batching) "
        "could handle many at once. Raise this to actually use the server's concurrency "
        "headroom; keep it within what the server's configured KV cache supports.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--backend", default="vllm", choices=["vllm", "sglang"])
    parser.add_argument(
        "--local-model-path",
        default=None,
        help="Local checkpoint directory (e.g. a merged fine-tuned model) instead of "
        "downloading --model from the Hub.",
    )
    parser.add_argument(
        "--skip-server-setup",
        action="store_true",
        help="Point at an already-running OpenAI-compatible server instead of letting "
        "BFCL launch/manage one.",
    )
    parser.add_argument("--vllm-endpoint", default="localhost")
    parser.add_argument("--vllm-port", default="8000")
    parser.add_argument("--result-dir", default=None)
    parser.add_argument("--score-dir", default=None)
    args = parser.parse_args()

    evaluator = BFCLEvaluator(
        model=args.model,
        test_category=args.test_category,
        local_model_path=args.local_model_path,
        num_gpus=args.num_gpus,
        num_threads=args.num_threads,
        gpu_memory_utilization=args.gpu_memory_utilization,
        backend=args.backend,
        skip_server_setup=args.skip_server_setup,
        vllm_endpoint=args.vllm_endpoint,
        vllm_port=args.vllm_port,
        result_dir=args.result_dir,
        score_dir=args.score_dir,
    )

    eval_dataset = evaluator.load_eval_dataset()
    print(f"Evaluating {evaluator.model} on {len(eval_dataset)} test file(s): {list(eval_dataset)}")
    predictions = evaluator.run_predictions(eval_dataset)
    evaluator.compute_metrics(predictions)


if __name__ == "__main__":
    main()
