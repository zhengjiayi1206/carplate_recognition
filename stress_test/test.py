#!/usr/bin/env python3
"""Stress test for stream API with detailed timing (Ack/Result events)."""

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


def encode_audio(file_path: str) -> str:
    """Encode audio file to base64."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_turn_files(session_dir: Path) -> list:
    """Get all turn files sorted by turn number."""
    audio_dir = session_dir / "audio"
    if not audio_dir.exists():
        return []
    
    turn_files = []
    for f in sorted(audio_dir.iterdir()):
        if f.name.startswith("turn_") and f.suffix == ".wav":
            turn_files.append(f)
    
    return turn_files


def test_session_turns_stream(
    session_id: str,
    session_dir: Path,
    base_url: str,
    model: str,
    timeout: int = 120,
) -> list:
    """Test all turns for a single session using stream API (sequential)."""
    results = []
    turn_files = get_turn_files(session_dir)
    
    if not turn_files:
        return results
    
    for turn_file in turn_files:
        turn_name = turn_file.name
        is_first_turn = (turn_name == "turn_000.wav")
        
        try:
            audio_base64 = encode_audio(str(turn_file))
            
            payload = {
                "session_id": session_id,
                "model": model,
                "audio_base64": audio_base64,
                "outputAudio": False,
            }
            
            # Use stream API
            total_start = time.perf_counter()
            events_with_timing = []
            
            with httpx.stream("POST", f"{base_url}/api/chatbox/audio/stream", json=payload, timeout=timeout) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            event_time_ms = int((time.perf_counter() - total_start) * 1000)
                            events_with_timing.append({
                                "time_ms": event_time_ms,
                                "event": data,
                            })
                        except json.JSONDecodeError:
                            pass
            
            total_time_ms = int((time.perf_counter() - total_start) * 1000)
            
            # Extract timing for each event
            ack_time_ms = None
            result_time_ms = None
            result_data = {}
            
            for item in events_with_timing:
                stage = item["event"].get("stage")
                if stage == "ack" and ack_time_ms is None:
                    ack_time_ms = item["time_ms"]
                elif stage == "result":
                    result_time_ms = item["time_ms"]
                    result_data = item["event"]
            
            server_latency = result_data.get("latency_ms", 0)
            agent_state = result_data.get("agent_state", {})
            
            # 延迟计算逻辑：有 Ack 用 Ack，没有 Ack 用 Result
            latency_ms = ack_time_ms if ack_time_ms is not None else result_time_ms
            
            results.append({
                "session_id": session_id,
                "turn": turn_name,
                "turn_type": "first_turn" if is_first_turn else "multi_turn",
                "status": "success",
                "total_time_ms": total_time_ms,
                "server_latency_ms": server_latency,
                "ack_time_ms": ack_time_ms,
                "result_time_ms": result_time_ms,
                "latency_ms": latency_ms,  # 最终延迟（Ack 或 Result）
                "ack_to_result_gap_ms": (result_time_ms - ack_time_ms) if ack_time_ms and result_time_ms else None,
                "car_plate": agent_state.get("car_plate", ""),
                "final_car_plate": agent_state.get("final_car_plate", ""),
                "assistant_reply": result_data.get("speech_text", "")[:100],
                "has_ack": ack_time_ms is not None,
                "events_count": len(events_with_timing),
            })
            
        except Exception as e:
            results.append({
                "session_id": session_id,
                "turn": turn_name,
                "turn_type": "first_turn" if is_first_turn else "multi_turn",
                "status": "error",
                "error": str(e)[:200],
            })
    
    return results


def run_single_round(
    sessions: list,
    base_url: str,
    model: str,
    concurrency_level: int,
    round_num: int,
    timeout: int = 120,
    launch_window: float = 1.0,
) -> tuple[list, dict]:
    """Run a single round of test."""
    results = []
    launch_interval = launch_window / len(sessions) if launch_window > 0 and sessions else 0
    
    with ThreadPoolExecutor(max_workers=len(sessions)) as executor:
        futures = {}
        round_launch_start = time.perf_counter()
        for idx, session_dir in enumerate(sessions):
            if launch_interval > 0:
                scheduled_at = round_launch_start + idx * launch_interval
                wait_seconds = scheduled_at - time.perf_counter()
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
            session_id_base = session_dir.name
            # Use unique session_id to avoid state from previous tests
            session_id = f"{session_id_base}_{concurrency_level}_{round_num}_{int(time.time() * 1000)}_{idx}"
            future = executor.submit(
                test_session_turns_stream,
                session_id,
                session_dir,
                base_url,
                model,
                timeout,
            )
            futures[future] = session_id
        
        for future in as_completed(futures):
            session_results = future.result()
            results.extend(session_results)
    
    # Calculate statistics
    first_turn_results = [r for r in results if r.get("turn_type") == "first_turn" and r.get("status") == "success"]
    multi_turn_results = [r for r in results if r.get("turn_type") == "multi_turn" and r.get("status") == "success"]
    
    # 提取延迟数据（有 Ack 用 Ack，没有 Ack 用 Result）
    first_turn_latencies = [r["latency_ms"] for r in first_turn_results if r.get("latency_ms")]
    multi_turn_latencies = [r["latency_ms"] for r in multi_turn_results if r.get("latency_ms")]
    
    # 额外的 Ack 和 Result 时间统计
    first_turn_ack_times = [r["ack_time_ms"] for r in first_turn_results if r.get("ack_time_ms")]
    first_turn_result_times = [r["result_time_ms"] for r in first_turn_results if r.get("result_time_ms")]
    multi_turn_ack_times = [r["ack_time_ms"] for r in multi_turn_results if r.get("ack_time_ms")]
    multi_turn_result_times = [r["result_time_ms"] for r in multi_turn_results if r.get("result_time_ms")]
    
    # 服务器延迟
    first_turn_server_latencies = [r["server_latency_ms"] for r in first_turn_results]
    multi_turn_server_latencies = [r["server_latency_ms"] for r in multi_turn_results]
    
    # Ack 到 Result 的间隔
    first_turn_ack_gaps = [r["ack_to_result_gap_ms"] for r in first_turn_results if r.get("ack_to_result_gap_ms")]
    multi_turn_ack_gaps = [r["ack_to_result_gap_ms"] for r in multi_turn_results if r.get("ack_to_result_gap_ms")]
    
    # 总时间
    first_turn_total_times = [r["total_time_ms"] for r in first_turn_results]
    multi_turn_total_times = [r["total_time_ms"] for r in multi_turn_results]
    
    success_count = len([r for r in results if r.get("status") == "success"])
    fail_count = len([r for r in results if r.get("status") in ["fail", "error", "timeout"]])
    
    def calc_percentile(data, percentile):
        """Calculate percentile with linear interpolation."""
        if not data:
            return 0
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n == 1:
            return sorted_data[0]
        idx = (n - 1) * percentile / 100
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        weight = idx - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    def calc_stats(data):
        """Calculate avg, p25, p50, p75, p95 for a list of numbers."""
        if not data:
            return {"avg": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0}
        return {
            "avg": round(sum(data) / len(data), 2),
            "p25": calc_percentile(data, 25),
            "p50": calc_percentile(data, 50),
            "p75": calc_percentile(data, 75),
            "p95": calc_percentile(data, 95),
        }
    
    stats = {
        "concurrency": concurrency_level,
        "round": round_num,
        "launch_window_seconds": launch_window,
        "launch_interval_seconds": round(launch_interval, 4),
        "total_sessions": len(sessions),
        "total_turns": len(results),
        "success_count": success_count,
        "fail_count": fail_count,
        
        # ===== 首轮统计 =====
        "first_turn_count": len([r for r in results if r.get("turn_type") == "first_turn"]),
        "first_turn_success": len(first_turn_results),
        
        # 首轮延迟统计（有 Ack 用 Ack，没有 Ack 用 Result）
        "first_turn_latency": calc_stats(first_turn_latencies),
        
        # 首轮 Ack 时间统计（仅针对有 Ack 的）
        "first_turn_ack_time": calc_stats(first_turn_ack_times),
        
        # 首轮 Result 时间统计
        "first_turn_result_time": calc_stats(first_turn_result_times),
        
        # 首轮服务器延迟
        "first_turn_server_latency": calc_stats(first_turn_server_latencies),
        
        # 首轮 Ack 到 Result 间隔
        "first_turn_ack_gap": calc_stats(first_turn_ack_gaps),
        
        # 首轮总时间
        "first_turn_total_time": calc_stats(first_turn_total_times),
        
        # ===== 多轮统计 =====
        "multi_turn_count": len([r for r in results if r.get("turn_type") == "multi_turn"]),
        "multi_turn_success": len(multi_turn_results),
        
        # 多轮延迟统计（有 Ack 用 Ack，没有 Ack 用 Result）
        "multi_turn_latency": calc_stats(multi_turn_latencies),
        
        # 多轮 Ack 时间统计（仅针对有 Ack 的）
        "multi_turn_ack_time": calc_stats(multi_turn_ack_times),
        
        # 多轮 Result 时间统计
        "multi_turn_result_time": calc_stats(multi_turn_result_times),
        
        # 多轮服务器延迟
        "multi_turn_server_latency": calc_stats(multi_turn_server_latencies),
        
        # 多轮 Ack 到 Result 间隔
        "multi_turn_ack_gap": calc_stats(multi_turn_ack_gaps),
        
        # 多轮总时间
        "multi_turn_total_time": calc_stats(multi_turn_total_times),
    }
    
    return results, stats


def run_concurrency_with_multiple_rounds(
    sessions: list,
    base_url: str,
    model: str,
    concurrency_level: int,
    num_rounds: int,
    timeout: int = 120,
    launch_window: float = 1.0,
) -> tuple[list, list]:
    """Run multiple rounds for a single concurrency level."""
    all_results = []
    all_stats = []
    
    for round_num in range(1, num_rounds + 1):
        print(f"  Round {round_num}/{num_rounds}...", end=" ", flush=True)
        start = time.time()
        
        start_idx = (round_num - 1) * concurrency_level
        end_idx = start_idx + concurrency_level
        round_sessions = sessions[start_idx:end_idx]
        
        results, stats = run_single_round(
            round_sessions,
            base_url,
            model,
            concurrency_level,
            round_num,
            timeout,
            launch_window,
        )
        duration = time.time() - start
        
        all_results.extend(results)
        all_stats.append(stats)
        print(f"OK ({stats['success_count']}/{stats['total_turns']} success) - {duration:.1f}s")
    
    return all_results, all_stats


def run_all_concurrency_tests(
    audio_dir: Path,
    base_url: str,
    model: str,
    concurrency_levels: list,
    num_rounds: int,
    timeout: int = 120,
    launch_window: float = 1.0,
    output_dir: str = "stress_test",
) -> dict:
    """Run tests for all concurrency levels."""
    all_sessions = sorted([d for d in audio_dir.iterdir() if d.is_dir() and (d / "audio" / "turn_000.wav").exists()])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Stream API Stress Test - First Turn & Multi-Turn Latency")
    print(f"{'='*80}")
    print(f"Total sessions available: {len(all_sessions)}")
    print(f"Rounds per concurrency: {num_rounds}")
    print(f"Timeout per request: {timeout}s")
    print(f"Launch window per round: {launch_window}s")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"Output directory: {output_path}")
    print(f"{'='*80}\n")
    
    all_summary = []
    
    for concurrency_level in concurrency_levels:
        total_sessions_needed = concurrency_level * num_rounds
        if total_sessions_needed > len(all_sessions):
            print(f"Error: Not enough sessions. Need {total_sessions_needed}, have {len(all_sessions)}")
            continue
        
        print(f"\n>>> Testing concurrency level: {concurrency_level} ({concurrency_level} sessions/round × {num_rounds} rounds)")
        print(f"{'-'*80}")
        
        sessions = all_sessions[:total_sessions_needed]
        
        start_time = time.time()
        results, stats_list = run_concurrency_with_multiple_rounds(
            sessions,
            base_url,
            model,
            concurrency_level,
            num_rounds,
            timeout,
            launch_window,
        )
        duration = time.time() - start_time
        
        total_success = sum(s["success_count"] for s in stats_list)
        total_turns = sum(s["total_turns"] for s in stats_list)
        
        print(f"\n  >>> Concurrency {concurrency_level} Summary:")
        print(f"      Total sessions: {total_sessions_needed}")
        print(f"      Total turns: {total_turns} | Success: {total_success}")
        print(f"      Total duration: {duration:.1f}s")
        
        # Save individual concurrency result
        concurrency_result = {
            "test_type": "stream_api_first_turn_and_multi_turn",
            "timestamp": datetime.now().isoformat(),
            "base_url": base_url,
            "model": model,
            "concurrency_level": concurrency_level,
            "num_rounds": num_rounds,
            "timeout": timeout,
            "launch_window_seconds": launch_window,
            "statistics_by_round": stats_list,
            "all_results": results,
        }
        
        concurrency_file = output_path / f"stream_stress_test_result_concurrency_{concurrency_level}.json"
        with open(concurrency_file, "w", encoding="utf-8") as f:
            json.dump(concurrency_result, f, ensure_ascii=False, indent=2)
        print(f"      Saved to: {concurrency_file}")
        
        # Add to summary
        all_summary.append({
            "concurrency": concurrency_level,
            "num_rounds": num_rounds,
            "total_sessions": total_sessions_needed,
            "total_turns": total_turns,
            "total_success": total_success,
            "total_duration_seconds": round(duration, 2),
            "statistics_by_round": stats_list,
        })
    
    # Save summary file
    summary_file = output_path / "stream_stress_test_result_summary.json"
    summary = {
        "test_type": "stream_api_summary",
        "timestamp": datetime.now().isoformat(),
        "base_url": base_url,
        "model": model,
        "concurrency_levels": concurrency_levels,
        "num_rounds": num_rounds,
        "timeout": timeout,
        "launch_window_seconds": launch_window,
        "summary_by_concurrency": all_summary,
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  >>> Summary saved to: {summary_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stream API Stress Test")
    parser.add_argument(
        "--audio-dir",
        "-d",
        default="stress_test/audio_data",
        help="Path to audio_data directory",
    )
    parser.add_argument(
        "--base-url",
        "-u",
        default="http://127.0.0.1:57300",
        help="API base URL",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="qwen3-omni",
        help="Model name",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="stress_test",
        help="Output directory for result files",
    )
    parser.add_argument(
        "--concurrency-levels",
        "-c",
        default="1,5,10,20,30,50,80,100",
        help="Comma-separated concurrency levels",
    )
    parser.add_argument(
        "--num-rounds",
        "-r",
        type=int,
        default=20,
        help="Number of rounds per concurrency level",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=120,
        help="Timeout per request in seconds",
    )
    parser.add_argument(
        "--launch-window",
        type=float,
        default=1.0,
        help="Seconds to evenly spread session starts within each round. Use 0 for simultaneous launch.",
    )
    
    args = parser.parse_args()
    
    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"Error: Directory not found: {audio_dir}")
        sys.exit(1)
    
    concurrency_levels = [int(x.strip()) for x in args.concurrency_levels.split(",")]
    
    run_all_concurrency_tests(
        audio_dir=audio_dir,
        base_url=args.base_url,
        model=args.model,
        concurrency_levels=concurrency_levels,
        num_rounds=args.num_rounds,
        timeout=args.timeout,
        launch_window=args.launch_window,
        output_dir=args.output_dir,
    )
    
    print(f"\n{'='*80}")
    print("All Tests Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
