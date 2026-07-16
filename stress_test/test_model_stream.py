#!/usr/bin/env python3
"""Stress test OpenAI-compatible streaming chat completions directly."""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


def parse_sse_data(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line or not line.startswith("data: "):
        return None
    data = line[6:].strip()
    if data == "[DONE]":
        return {"done": True}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {"parse_error": data[:500]}


def extract_delta_text(data: dict[str, Any]) -> str:
    text_parts: list[str] = []
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text"):
                    text_parts.append(str(item["text"]))
        message = choice.get("message") or {}
        message_content = message.get("content")
        if isinstance(message_content, str):
            text_parts.append(message_content)
    return "".join(text_parts)


def run_one_request(
    *,
    request_id: int,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    start = time.perf_counter()
    first_chunk_ms: int | None = None
    first_text_ms: int | None = None
    chunk_count = 0
    text_chunk_count = 0
    output_text_parts: list[str] = []
    error = ""
    status_code = 0

    try:
        with httpx.stream("POST", url, json=payload, timeout=timeout) as response:
            status_code = response.status_code
            response.raise_for_status()
            for raw_line in response.iter_lines():
                now_ms = int((time.perf_counter() - start) * 1000)
                data = parse_sse_data(raw_line)
                if data is None:
                    continue
                if data.get("done"):
                    break
                if first_chunk_ms is None:
                    first_chunk_ms = now_ms
                if data.get("parse_error"):
                    error = f"bad sse json: {data['parse_error']}"
                    break
                chunk_count += 1
                delta_text = extract_delta_text(data)
                if delta_text:
                    if first_text_ms is None:
                        first_text_ms = now_ms
                    text_chunk_count += 1
                    output_text_parts.append(delta_text)
    except Exception as exc:
        error = str(exc)[:500]

    total_ms = int((time.perf_counter() - start) * 1000)
    output_text = "".join(output_text_parts)
    char_count = len(output_text)
    generation_ms = max(0, total_ms - (first_text_ms or first_chunk_ms or total_ms))
    chars_per_second = round(char_count / (generation_ms / 1000), 2) if generation_ms > 0 else 0
    chunks_per_second = round(text_chunk_count / (generation_ms / 1000), 2) if generation_ms > 0 else 0

    return {
        "request_id": request_id,
        "status": "error" if error else "success",
        "status_code": status_code,
        "error": error,
        "first_chunk_ms": first_chunk_ms,
        "first_text_ms": first_text_ms,
        "total_ms": total_ms,
        "chunk_count": chunk_count,
        "text_chunk_count": text_chunk_count,
        "char_count": char_count,
        "chars_per_second": chars_per_second,
        "chunks_per_second": chunks_per_second,
        "output_preview": output_text[:200],
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * p / 100
    lower = int(idx)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def stats(values: list[float]) -> dict[str, float]:
    values = [value for value in values if value is not None]
    if not values:
        return {"avg": 0, "min": 0, "p50": 0, "p75": 0, "p95": 0, "p99": 0, "max": 0}
    return {
        "avg": round(statistics.mean(values), 2),
        "min": round(min(values), 2),
        "p50": round(percentile(values, 50), 2),
        "p75": round(percentile(values, 75), 2),
        "p95": round(percentile(values, 95), 2),
        "p99": round(percentile(values, 99), 2),
        "max": round(max(values), 2),
    }


def run_test(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Direct Model Streaming Stress Test")
    print("=" * 80)
    print(f"Base URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Launch window: {args.launch_window}s")
    print(f"Max tokens: {args.max_tokens}")
    print("=" * 80)

    results: list[dict[str, Any]] = []
    launch_interval = args.launch_window / args.concurrency if args.launch_window > 0 else 0
    test_start = time.perf_counter()
    launch_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for idx in range(args.concurrency):
            if launch_interval > 0:
                scheduled_at = launch_start + idx * launch_interval
                wait_seconds = scheduled_at - time.perf_counter()
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
            futures.append(
                executor.submit(
                    run_one_request,
                    request_id=idx,
                    base_url=args.base_url,
                    model=args.model,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
            )

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            marker = "OK" if result["status"] == "success" else "ERR"
            print(
                f"[{marker}] id={result['request_id']} "
                f"ttft={result['first_text_ms']}ms total={result['total_ms']}ms "
                f"chars={result['char_count']} cps={result['chars_per_second']}"
            )

    total_duration = round(time.perf_counter() - test_start, 2)
    success = [item for item in results if item["status"] == "success"]
    failed = [item for item in results if item["status"] != "success"]

    summary = {
        "test_type": "direct_model_stream",
        "timestamp": datetime.now().isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "launch_window_seconds": args.launch_window,
        "launch_interval_seconds": round(launch_interval, 4),
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "total_duration_seconds": total_duration,
        "success_count": len(success),
        "failed_count": len(failed),
        "first_chunk_ms": stats([item["first_chunk_ms"] for item in success if item["first_chunk_ms"] is not None]),
        "first_text_ms": stats([item["first_text_ms"] for item in success if item["first_text_ms"] is not None]),
        "total_ms": stats([item["total_ms"] for item in success]),
        "chars_per_second": stats([item["chars_per_second"] for item in success]),
        "chunks_per_second": stats([item["chunks_per_second"] for item in success]),
        "results": sorted(results, key=lambda item: item["request_id"]),
    }

    output_file = output_dir / f"model_stream_stress_{args.concurrency}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSummary")
    print("-" * 80)
    print(f"Success: {len(success)}/{len(results)}")
    print(f"First text latency: {summary['first_text_ms']}")
    print(f"Total latency: {summary['total_ms']}")
    print(f"Chars/sec: {summary['chars_per_second']}")
    print(f"Saved: {output_file}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test direct OpenAI-compatible streaming model API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5440/v1", help="OpenAI-compatible API base URL.")
    parser.add_argument("--model", default="qwen3-omni", help="Model name.")
    parser.add_argument("--concurrency", "-c", type=int, default=50, help="Number of concurrent requests in one round.")
    parser.add_argument("--launch-window", type=float, default=1.0, help="Seconds to spread request starts. Use 0 for simultaneous.")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per request.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout per request in seconds.")
    parser.add_argument("--output-dir", default="stress_test", help="Directory for result JSON.")
    parser.add_argument(
        "--prompt",
        default="请用一段话介绍一下中国新能源汽车号牌的特点。",
        help="Prompt used for every request.",
    )
    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
