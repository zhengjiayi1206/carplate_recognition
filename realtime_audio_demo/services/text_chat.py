import json
import logging
import time
from typing import Any

from realtime_audio_demo.config import QWEN_API_BASE
from realtime_audio_demo.services.qwen import build_text_payload, extract_model_output, post_json

logger = logging.getLogger("uvicorn.error")


async def request_text_completion(
    *,
    model: str,
    text: str,
    prompt: str,
    history: list[dict[str, Any]],
    max_tokens: int,
    output_audio: bool = False,
    response_format: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    payload = build_text_payload(
        model,
        text,
        prompt,
        history=history,
        max_tokens=max_tokens,
        modalities=["text", "audio"] if output_audio else ["text"],
        response_format=response_format,
    )

    start = time.perf_counter()
    try:
        response = await post_json(f"{QWEN_API_BASE}/chat/completions", payload)
    except Exception as exc:
        return {"message": str(exc)}, 502

    latency_ms = int((time.perf_counter() - start) * 1000)
    ttft_ms = response.ttft_ms

    logger.info(
        "text_chat latency=%dms ttft=%s history=%d",
        latency_ms,
        ttft_ms if ttft_ms is not None else "n/a",
        len(history),
    )

    if response.status_code >= 400:
        return (
            {
                "message": response.text[:2000],
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
            response.status_code,
        )

    try:
        data = response.json()
    except json.JSONDecodeError as exc:
        return {"message": f"bad upstream json: {exc}", "latency_ms": latency_ms}, 502

    parsed = extract_model_output(data)
    return (
        {
            "text": parsed["text"],
            "audio_data_url": parsed["audio_data_url"],
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
        },
        200,
    )
