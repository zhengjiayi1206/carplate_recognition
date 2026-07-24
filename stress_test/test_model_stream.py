#!/usr/bin/env python3
"""Stress test /api/chatbox/audio/stream first reply and total latency."""

import argparse
import base64
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


def encode_audio(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_sse_data(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    if line == "data: [DONE]":
        return {"done": True}
    if not line.startswith("data:"):
        return None
    raw = line[5:].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"parse_error": raw[:500]}


def turn_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    try:
        return int(stem.split("_", 1)[1]), path.name
    except (IndexError, ValueError):
        return 999999, path.name


def get_turn_files(session_dir: Path, max_turns: int | None) -> list[Path]:
    audio_dir = session_dir / "audio"
    search_dir = audio_dir if audio_dir.exists() else session_dir
    turns = sorted(search_dir.glob("turn_*.wav"), key=turn_sort_key)
    if max_turns is not None and max_turns > 0:
        return turns[:max_turns]
    return turns


def discover_sessions(audio_dir: Path) -> list[Path]:
    if get_turn_files(audio_dir, max_turns=1):
        return [audio_dir]
    return sorted(
        path
        for path in audio_dir.iterdir()
        if path.is_dir() and get_turn_files(path, max_turns=1)
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], 2)
    idx = (len(sorted_values) - 1) * p / 100
    lower = int(idx)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = idx - lower
    return round(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight, 2)


def stats(values: list[float | int | None]) -> dict[str, float | int]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"count": 0, "avg": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0}
    return {
        "count": len(clean),
        "avg": round(statistics.mean(clean), 2),
        "p25": percentile(clean, 25),
        "p50": percentile(clean, 50),
        "p75": percentile(clean, 75),
        "p95": percentile(clean, 95),
    }


def request_audio_stream_turn(
    *,
    request_id: str,
    session_id: str,
    turn_file: Path,
    turn_index: int,
    base_url: str,
    model: str,
    timeout: float,
    output_audio: bool,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/chatbox/audio/stream"
    payload = {
        "session_id": session_id,
        "model": model,
        "audio_base64": encode_audio(turn_file),
        "outputAudio": output_audio,
    }

    started = time.perf_counter()
    status_code = 0
    error = ""
    first_event_ms: int | None = None
    first_ack_ms: int | None = None
    first_ack_text = ""
    last_ack_text = ""
    result_ms: int | None = None
    result_event: dict[str, Any] = {}
    event_count = 0
    ack_count = 0

    try:
        with httpx.stream("POST", url, json=payload, timeout=timeout) as response:
            status_code = response.status_code
            response.raise_for_status()
            for raw_line in response.iter_lines():
                now_ms = int((time.perf_counter() - started) * 1000)
                data = parse_sse_data(raw_line)
                if data is None:
                    continue
                if data.get("done"):
                    break
                if data.get("parse_error"):
                    error = f"bad sse json: {data['parse_error']}"
                    break

                event_count += 1
                if first_event_ms is None:
                    first_event_ms = now_ms

                stage = str(data.get("stage") or "")
                if stage == "ack":
                    ack_text = str(data.get("speech_text") or "")
                    ack_count += 1
                    last_ack_text = ack_text
                    if first_ack_ms is None:
                        first_ack_ms = now_ms
                        first_ack_text = ack_text
                elif stage == "result":
                    result_ms = now_ms
                    result_event = data
    except Exception as exc:
        error = str(exc)[:500]

    total_ms = int((time.perf_counter() - started) * 1000)
    first_reply_ms = first_ack_ms if first_ack_ms is not None else result_ms
    agent_state = result_event.get("agent_state") if isinstance(result_event.get("agent_state"), dict) else {}

    return {
        "request_id": request_id,
        "session_id": session_id,
        "turn": turn_file.name,
        "turn_index": turn_index,
        "turn_type": "first_turn" if turn_index == 0 else "multi_turn",
        "status": "success" if not error and result_ms is not None else "error",
        "status_code": status_code,
        "error": error,
        "first_event_ms": first_event_ms,
        "first_ack_ms": first_ack_ms,
        "first_reply_ms": first_reply_ms,
        "result_ms": result_ms,
        "total_ms": total_ms,
        "server_latency_ms": result_event.get("latency_ms"),
        "ack_to_result_gap_ms": result_ms - first_ack_ms if result_ms is not None and first_ack_ms is not None else None,
        "event_count": event_count,
        "ack_count": ack_count,
        "first_ack_text": first_ack_text[:200],
        "last_ack_text": last_ack_text[:200],
        "result_speech_text": str(result_event.get("speech_text") or "")[:200],
        "car_plate": agent_state.get("car_plate", ""),
        "final_car_plate": agent_state.get("final_car_plate", ""),
    }


def run_session_turns(
    *,
    request_id_prefix: str,
    session_dir: Path,
    session_id: str,
    base_url: str,
    model: str,
    timeout: float,
    output_audio: bool,
    max_turns: int | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for turn_index, turn_file in enumerate(get_turn_files(session_dir, max_turns=max_turns)):
        result = request_audio_stream_turn(
            request_id=f"{request_id_prefix}_turn_{turn_index:03d}",
            session_id=session_id,
            turn_file=turn_file,
            turn_index=turn_index,
            base_url=base_url,
            model=model,
            timeout=timeout,
            output_audio=output_audio,
        )
        results.append(result)
    return results


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    success = [item for item in results if item.get("status") == "success"]
    failed = [item for item in results if item.get("status") != "success"]
    first_turn = [item for item in success if item.get("turn_type") == "first_turn"]
    multi_turn = [item for item in success if item.get("turn_type") == "multi_turn"]

    def group_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "success_count": len(items),
            "first_reply_ms": stats([item.get("first_reply_ms") for item in items]),
            "first_ack_ms": stats([item.get("first_ack_ms") for item in items]),
            "result_ms": stats([item.get("result_ms") for item in items]),
            "total_ms": stats([item.get("total_ms") for item in items]),
            "server_latency_ms": stats([item.get("server_latency_ms") for item in items]),
            "ack_to_result_gap_ms": stats([item.get("ack_to_result_gap_ms") for item in items]),
            "ack_count": stats([item.get("ack_count") for item in items]),
        }

    return {
        "total_turns": len(results),
        "success_count": len(success),
        "failed_count": len(failed),
        "first_turn": group_summary(first_turn),
        "multi_turn": group_summary(multi_turn),
    }


def print_group(label: str, data: dict[str, Any]) -> None:
    print(f"{label}: success={data['success_count']}")
    print(f"  first_reply_ms: {data['first_reply_ms']}")
    print(f"  total_ms:       {data['total_ms']}")


def run_test(args: argparse.Namespace) -> dict[str, Any]:
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_dir.exists():
        raise FileNotFoundError(f"audio dir not found: {audio_dir}")

    sessions = discover_sessions(audio_dir)
    if not sessions:
        raise FileNotFoundError(f"no sessions with turn_*.wav found under: {audio_dir}")

    total_session_runs = args.concurrency * args.rounds
    launch_interval = args.launch_window / args.concurrency if args.launch_window > 0 and args.concurrency > 0 else 0
    all_results: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []

    print("=" * 80)
    print("Chatbox Audio Stream Stress Test")
    print("=" * 80)
    print(f"URL: {args.base_url.rstrip('/')}/api/chatbox/audio/stream")
    print(f"Model: {args.model}")
    print(f"Audio dir: {audio_dir}")
    print(f"Sessions discovered: {len(sessions)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Rounds: {args.rounds}")
    print(f"Launch window: {args.launch_window}s")
    print(f"Output audio: {args.output_audio}")
    print("=" * 80)

    test_started = time.perf_counter()
    for round_index in range(args.rounds):
        round_started = time.perf_counter()
        futures = []
        round_results: list[dict[str, Any]] = []
        launch_started = time.perf_counter()

        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for worker_index in range(args.concurrency):
                if launch_interval > 0:
                    scheduled_at = launch_started + worker_index * launch_interval
                    wait_seconds = scheduled_at - time.perf_counter()
                    if wait_seconds > 0:
                        time.sleep(wait_seconds)

                run_index = round_index * args.concurrency + worker_index
                session_dir = sessions[run_index % len(sessions)]
                session_id = (
                    f"{session_dir.name}_stream_r{round_index + 1}_w{worker_index}_"
                    f"{int(time.time() * 1000)}"
                )
                futures.append(
                    executor.submit(
                        run_session_turns,
                        request_id_prefix=f"r{round_index + 1}_w{worker_index}",
                        session_dir=session_dir,
                        session_id=session_id,
                        base_url=args.base_url,
                        model=args.model,
                        timeout=args.timeout,
                        output_audio=args.output_audio,
                        max_turns=args.max_turns,
                    )
                )

            for future in as_completed(futures):
                session_results = future.result()
                round_results.extend(session_results)
                all_results.extend(session_results)

        round_summary = summarize_results(round_results)
        round_summary["round"] = round_index + 1
        round_summary["duration_seconds"] = round(time.perf_counter() - round_started, 2)
        round_summaries.append(round_summary)
        print(
            f"[round {round_index + 1}/{args.rounds}] "
            f"success={round_summary['success_count']}/{round_summary['total_turns']} "
            f"duration={round_summary['duration_seconds']}s"
        )

    summary_stats = summarize_results(all_results)
    summary = {
        "test_type": "chatbox_audio_stream",
        "timestamp": datetime.now().isoformat(),
        "base_url": args.base_url,
        "endpoint": "/api/chatbox/audio/stream",
        "model": args.model,
        "audio_dir": str(audio_dir),
        "sessions_discovered": len(sessions),
        "session_runs": total_session_runs,
        "concurrency": args.concurrency,
        "rounds": args.rounds,
        "launch_window_seconds": args.launch_window,
        "timeout": args.timeout,
        "output_audio": args.output_audio,
        "max_turns": args.max_turns,
        "total_duration_seconds": round(time.perf_counter() - test_started, 2),
        "summary": summary_stats,
        "round_summaries": round_summaries,
        "results": all_results,
    }

    output_file = output_dir / f"audio_stream_latency_{args.concurrency}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSummary")
    print("-" * 80)
    print_group("First turn", summary_stats["first_turn"])
    print_group("Multi turn", summary_stats["multi_turn"])
    print(f"Saved: {output_file}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test /api/chatbox/audio/stream latency.")
    parser.add_argument("--audio-dir", default="stress_test/audio_data", help="Directory containing session/audio/turn_*.wav files.")
    parser.add_argument("--base-url", default="http://127.0.0.1:57300", help="Application API base URL, without endpoint path.")
    parser.add_argument("--model", default="qwen3-omni", help="Model name sent to the app API.")
    parser.add_argument("--concurrency", "-c", type=int, default=1, help="Parallel sessions per round.")
    parser.add_argument("--rounds", "-r", type=int, default=1, help="Number of rounds.")
    parser.add_argument("--launch-window", type=float, default=1.0, help="Seconds to spread session starts in each round.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout per streamed turn request.")
    parser.add_argument("--output-dir", default="stress_test", help="Directory for result JSON files.")
    parser.add_argument("--max-turns", type=int, default=0, help="Limit turns per session. 0 means all turns.")
    parser.add_argument("--output-audio", action="store_true", help="Request TTS audio too. This can distort ack latency.")
    args = parser.parse_args()

    try:
        run_test(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
