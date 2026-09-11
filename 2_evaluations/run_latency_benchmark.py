"""Concurrency-swept load test against a locally served checkpoint.

Measures throughput, time-to-first-token (TTFT), and inter-token latency (ITL) at each
concurrency level, streaming completions so per-token timing is measured directly rather
than estimated from total request latency. Shares its concurrency sweep with
3_optimizations/benchmark_serving.py so the two are directly comparable.

Usage:
    python run_latency_benchmark.py --endpoint http://localhost:8000/v1 --model my-checkpoint \
        --prompts-file ../0_data/data/test.jsonl --concurrency 1 8 16 24 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


def load_prompts(path: Path, limit: int) -> list[list[dict[str, str]]]:
    prompts = []
    with path.open() as f:
        for line in f:
            example = json.loads(line)
            messages = [{"role": "system", "content": example["system"]}]
            for turn in example["turns"]:
                if turn["role"] == "user":
                    messages.append({"role": "user", "content": turn["content"]})
                    break
            prompts.append(messages)
            if len(prompts) >= limit:
                break
    return prompts


async def run_one(
    client: AsyncOpenAI, model: str, messages: list[dict[str, str]], enable_thinking: bool
) -> dict[str, float]:
    start = time.perf_counter()
    first_token_time: float | None = None
    token_times: list[float] = []
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        stream=True,
        # Qwen3 is a hybrid reasoning model — leaving thinking mode on would inflate
        # TTFT/throughput with chain-of-thought generation instead of measuring the
        # actual answer latency this benchmark exists to capture. Ignored by non-Qwen3
        # models. See vLLM's chat_template_kwargs extra_body convention.
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    async for chunk in stream:
        if not chunk.choices or chunk.choices[0].delta.content is None:
            continue
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now
        token_times.append(now)
    end = time.perf_counter()
    n_tokens = len(token_times)
    ttft = (first_token_time - start) if first_token_time else (end - start)
    itl = (token_times[-1] - token_times[0]) / (n_tokens - 1) if n_tokens > 1 else 0.0
    return {"ttft": ttft, "itl": itl, "n_tokens": n_tokens, "duration": end - start}


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return quantiles[min(max(int(pct) - 1, 0), 98)]


async def run_concurrency_level(
    endpoint: str,
    api_key: str,
    model: str,
    prompts: list[list[dict[str, str]]],
    concurrency: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    client = AsyncOpenAI(base_url=endpoint, api_key=api_key)
    batch = [prompts[i % len(prompts)] for i in range(concurrency)]
    start = time.perf_counter()
    results = await asyncio.gather(
        *(run_one(client, model, m, enable_thinking) for m in batch)
    )
    wall_time = time.perf_counter() - start

    total_tokens = sum(r["n_tokens"] for r in results)
    ttfts = sorted(r["ttft"] for r in results)
    itls = sorted(r["itl"] for r in results if r["n_tokens"] > 1)

    return {
        "concurrency": concurrency,
        "throughput_tok_per_s": total_tokens / wall_time if wall_time else 0.0,
        "ttft_p50": _percentile(ttfts, 50),
        "ttft_p90": _percentile(ttfts, 90),
        "ttft_p99": _percentile(ttfts, 99),
        "itl_p50": _percentile(itls, 50),
        "itl_p90": _percentile(itls, 90),
        "itl_p99": _percentile(itls, 99),
    }


async def main_async(args: argparse.Namespace) -> None:
    prompts = load_prompts(args.prompts_file, limit=max(args.concurrency))
    results = []
    for c in args.concurrency:
        result = await run_concurrency_level(
            args.endpoint, args.api_key, args.model, prompts, c, args.enable_thinking
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 16, 24, 32])
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--output", type=Path, default=Path("results/latency_benchmark.json"))
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Leave off (default) to measure non-thinking-mode latency — the production-relevant number. Ignored by non-Qwen3 models.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
