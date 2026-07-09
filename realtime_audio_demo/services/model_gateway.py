import asyncio
import time
from typing import Any, Callable

from realtime_audio_demo.config import QWEN_API_BASE
from realtime_audio_demo.services.interfaces import ChatModel, CompletionResult
from realtime_audio_demo.services.qwen import (
    build_chat_payload,
    extract_model_output,
    post_json,
    stream_json_sync,
)
from realtime_audio_demo.services.text_chat import request_text_completion


class OpenAICompatibleQwenModel(ChatModel):
    async def complete_text(
        self,
        *,
        model: str,
        text: str,
        prompt: str,
        history: list[dict[str, Any]],
        max_tokens: int,
        output_audio: bool = False,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        return await request_text_completion(
            model=model,
            text=text,
            prompt=prompt,
            history=history,
            max_tokens=max_tokens,
            output_audio=output_audio,
            response_format=response_format,
        )

    async def complete_audio(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        prompt: str,
        history: list[dict[str, Any]],
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
        turn_instruction: str | None = None,
    ) -> CompletionResult:
        payload = build_chat_payload(
            model,
            wav_bytes,
            prompt,
            history=history,
            max_tokens=max_tokens,
            modalities=["text"],
            response_format=response_format,
            turn_instruction=turn_instruction,
        )
        start = time.perf_counter()
        response = await post_json(f"{QWEN_API_BASE}/chat/completions", payload)
        latency_ms = int((time.perf_counter() - start) * 1000)
        if response.status_code >= 400:
            return CompletionResult(
                text=response.text[:2000],
                audio_data_url=None,
                raw_response={"status_code": response.status_code, "message": response.text[:2000]},
                latency_ms=latency_ms,
                ttft_ms=response.ttft_ms,
            )

        data = response.json()
        parsed = extract_model_output(data)
        return CompletionResult(
            text=parsed["text"],
            audio_data_url=parsed["audio_data_url"],
            raw_response=data,
            latency_ms=latency_ms,
            ttft_ms=response.ttft_ms,
        )

    async def stream_audio(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        prompt: str,
        history: list[dict[str, Any]],
        max_tokens: int,
        response_format: dict[str, Any] | None,
        push: Callable[[dict[str, Any]], None],
        turn_instruction: str | None = None,
    ) -> None:
        payload = build_chat_payload(
            model,
            wav_bytes,
            prompt,
            history=history,
            max_tokens=max_tokens,
            modalities=["text"],
            response_format=response_format,
            turn_instruction=turn_instruction,
        )
        payload = {**payload, "stream": True}
        await asyncio.to_thread(stream_json_sync, f"{QWEN_API_BASE}/chat/completions", payload, push)


model_gateway = OpenAICompatibleQwenModel()
