#!/usr/bin/env python3
"""Trace every SSE event returned by /api/chatbox/audio/stream for one audio file."""

import argparse
import base64
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


def encode_audio(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_sse_line(line: str) -> tuple[str, dict[str, Any] | None]:
    line = line.strip()
    if not line:
        return "", None
    if line == "data: [DONE]":
        return "[DONE]", {"done": True}
    if not line.startswith("data:"):
        return line, None
    raw = line[5:].strip()
    try:
        return raw, json.loads(raw)
    except json.JSONDecodeError:
        return raw, {"parse_error": raw}


def short_text(value: Any, limit: int = 500) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def trace_audio_stream(args: argparse.Namespace) -> dict[str, Any]:
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    base_url = args.base_url.rstrip("/")
    url = f"{base_url}/api/chatbox/audio/stream"
    session_id = args.session_id or f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    payload = {
        "session_id": session_id,
        "model": args.model,
        "audio_base64": encode_audio(audio_path),
        "outputAudio": args.output_audio,
    }

    print("=" * 80)
    print("Audio Stream SSE Trace")
    print("=" * 80)
    print(f"URL:        {url}")
    print(f"Audio:      {audio_path}")
    print(f"Session ID: {session_id}")
    print(f"Model:      {args.model}")
    print(f"OutputAudio:{args.output_audio}")
    print("=" * 80)

    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    previous_ms = 0
    status_code = 0

    with httpx.stream("POST", url, json=payload, timeout=args.timeout) as response:
        status_code = response.status_code
        response.raise_for_status()
        for line in response.iter_lines():
            raw, data = parse_sse_line(line)
            if data is None:
                continue

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            gap_ms = elapsed_ms - previous_ms
            previous_ms = elapsed_ms

            event = {
                "index": len(events) + 1,
                "elapsed_ms": elapsed_ms,
                "gap_ms": gap_ms,
                "raw": raw,
                "event": data,
                "stage": data.get("stage"),
                "speech_text": data.get("speech_text"),
                "text": data.get("text"),
                "latency_ms": data.get("latency_ms"),
                "audio_data_url_present": bool(data.get("audio_data_url")),
            }
            events.append(event)

            if data.get("done"):
                print(f"[{event['index']:03d}] +{elapsed_ms:6d}ms gap={gap_ms:5d}ms DONE")
                break

            stage = data.get("stage") or data.get("type") or ""
            speech_text = short_text(data.get("speech_text"))
            text = short_text(data.get("text"))
            print(f"[{event['index']:03d}] +{elapsed_ms:6d}ms gap={gap_ms:5d}ms stage={stage}")
            if speech_text:
                print(f"      speech_text: {speech_text}")
            if text and text != speech_text:
                print(f"      text: {text}")
            if args.print_raw:
                print(f"      raw: {short_text(raw, 1200)}")

    total_ms = int((time.perf_counter() - started) * 1000)
    ack_events = [item for item in events if item.get("stage") == "ack"]
    result_events = [item for item in events if item.get("stage") == "result"]
    done_events = [item for item in events if item.get("event", {}).get("done")]
    first_ack = next((item for item in events if item.get("stage") == "ack"), None)
    result = next((item for item in events if item.get("stage") == "result"), None)
    result_agent_state = result.get("event", {}).get("agent_state", {}) if result else {}
    runtime_version = result_agent_state.get("plate_agent_runtime") if isinstance(result_agent_state, dict) else None
    summary = {
        "test_type": "audio_stream_sse_trace",
        "timestamp": datetime.now().isoformat(),
        "url": url,
        "audio_path": str(audio_path),
        "session_id": session_id,
        "model": args.model,
        "output_audio": args.output_audio,
        "status_code": status_code,
        "total_ms": total_ms,
        "event_count": len(events),
        "ack_count": len(ack_events),
        "result_count": len(result_events),
        "done_count": len(done_events),
        "plate_agent_runtime": runtime_version,
        "first_ack_ms": first_ack.get("elapsed_ms") if first_ack else None,
        "result_ms": result.get("elapsed_ms") if result else None,
        "ack_texts": [
            {
                "index": item["index"],
                "elapsed_ms": item["elapsed_ms"],
                "gap_ms": item["gap_ms"],
                "length": len(str(item.get("speech_text") or "")),
                "speech_text": item.get("speech_text"),
            }
            for item in ack_events
        ],
        "events": events,
    }

    output_path = Path(args.output)
    if output_path.is_dir() or not output_path.suffix:
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f"audio_stream_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 80)
    print(f"Total:        {total_ms}ms")
    print(f"Events:       {len(events)}")
    print(f"Ack events:   {len(ack_events)}")
    print(f"Result events:{len(result_events)}")
    print(f"Done events:  {len(done_events)}")
    print(f"Agent runtime:{runtime_version or '<missing>'}")
    print(f"First ack:    {summary['first_ack_ms']}ms")
    print(f"Result:       {summary['result_ms']}ms")
    if len(ack_events) <= 1:
        print("WARNING: only one or zero ack events. The server may not be running the latest delta-ack plate_agent.py.")
    if ack_events:
        print("Ack detail:")
        for item in ack_events[:50]:
            text = str(item.get("speech_text") or "")
            print(f"  [{item['index']:03d}] +{item['elapsed_ms']}ms len={len(text)} text={short_text(text, 120)}")
    print(f"Saved JSON:   {output_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace /api/chatbox/audio/stream SSE events for one audio file.")
    parser.add_argument("audio_path", help="Path to one wav audio file.")
    parser.add_argument("--base-url", default="http://127.0.0.1:55785", help="App base URL.")
    parser.add_argument("--model", default="qwen3-omni", help="Model name.")
    parser.add_argument("--session-id", default="", help="Optional session id. Empty means auto-generate a fresh first-turn session.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")
    parser.add_argument("--output", default="stress_test", help="Output JSON file or directory.")
    parser.add_argument("--output-audio", action="store_true", help="Request TTS audio too.")
    parser.add_argument("--print-raw", action="store_true", help="Print raw SSE JSON data.")
    args = parser.parse_args()
    trace_audio_stream(args)


if __name__ == "__main__":
    main()
