"""Patches the installed bfcl-eval package's QwenHandler on disk to append the
enable_thinking=False "no-think" stub to every prompt it builds.

Must run as a standalone step, after bfcl-eval is installed and before `bfcl generate`/
`bfcl evaluate` run -- NOT an in-process monkeypatch. run_predictions() in
run_bfcl_eval.py shells out to the real `bfcl` CLI via subprocess.run(), which starts a
brand-new Python interpreter that re-imports bfcl_eval fresh from disk; a monkeypatch
applied in this repo's own process never reaches that child process. Confirmed the hard
way: an in-process monkeypatch of QwenHandler._format_prompt in run_bfcl_eval.py had zero
effect on a real eval run (same 0/16 live_parallel accuracy before and after).

Root cause: QwenHandler._format_prompt (bfcl_eval/model_handler/local_inference/qwen.py)
hand-builds the raw-completions prompt sent to /v1/completions. Its own docstring quotes
Qwen3's real chat template, which appends "<think>\n\n</think>\n\n" right after
"<|im_start|>assistant\n" when enable_thinking=False -- but the Python implementation
below that docstring never does this. Since /v1/completions has no chat-template stage at
all (confirmed against base_oss_handler.py's _query_prompting, the only query path local
OSS handlers have), nothing else suppresses Qwen3's default reasoning behavior, and the
model emits an unclosed leading <think> that breaks BFCL's parser -- confirmed against a
real BFCL_v3_live_parallel_result.json entry: "<think>\n[get_current_weather(...)]"
(a correct call, made unparseable by the leaked tag).

Usage (after `pip install bfcl-eval==...`, before `bfcl generate`):
    python patch_bfcl_qwen_handler.py
"""

from __future__ import annotations

from bfcl_eval.model_handler.local_inference import qwen

OLD_TAIL = '        formatted_prompt += "<|im_start|>assistant\\n"\n        return formatted_prompt'
NEW_TAIL = '        formatted_prompt += "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"\n        return formatted_prompt'


def main() -> None:
    path = qwen.__file__
    src = open(path).read()
    if OLD_TAIL not in src:
        raise RuntimeError(
            f"QwenHandler._format_prompt's expected tail not found in {path} -- "
            "bfcl-eval's source may have changed since this patch was written "
            "(pinned version: see bfcl-eval==... in this Job's pip install / tox.ini)."
        )
    src = src.replace(OLD_TAIL, NEW_TAIL, 1)
    open(path, "w").write(src)
    print(f"[patch_bfcl_qwen_handler] patched {path}: appends enable_thinking=False stub")


if __name__ == "__main__":
    main()
