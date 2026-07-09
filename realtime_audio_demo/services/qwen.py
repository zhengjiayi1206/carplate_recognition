import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from realtime_audio_demo.config import (
    MAX_HISTORY_TURNS,
    QWEN_MODALITIES,
    QWEN_SPEAKER,
    REQUEST_TIMEOUT,
    TARGET_SAMPLE_RATE,
    resolved_provider,
)

logger = logging.getLogger("uvicorn.error")

_http_client: httpx.Client | None = None


def _get_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=20),
        )
    return _http_client


@dataclass
class JsonResponseData:
    status_code: int
    text: str
    ttft_ms: Optional[int] = None

    def json(self) -> Any:
        return json.loads(self.text)


async def post_json(url: str, payload: dict[str, Any]) -> JsonResponseData:
    return await asyncio.to_thread(post_json_sync, url, payload)


def post_json_sync(url: str, payload: dict[str, Any]) -> JsonResponseData:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    client = _get_client()
    try:
        response = client.post(
            url,
            content=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        return JsonResponseData(
            status_code=response.status_code,
            text=response.text,
            ttft_ms=int(response.elapsed.total_seconds() * 1000),
        )
    except httpx.HTTPStatusError as exc:
        return JsonResponseData(
            status_code=exc.response.status_code,
            text=exc.response.text,
        )
    except Exception as exc:
        return JsonResponseData(
            status_code=502,
            text=f"upstream request failed: {exc}",
        )


def stream_json_sync(url: str, payload: dict[str, Any], push: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    client = _get_client()
    req_start = time.perf_counter()
    try:
        with client.stream(
            "POST",
            url,
            content=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        ) as response:
            if response.status_code >= 400:
                push({
                    "type": "error",
                    "status_code": response.status_code,
                    "message": response.read().decode("utf-8", errors="replace")[:2000],
                })
                return

            first_token = True
            for raw_line in response.iter_lines():
                line = raw_line.strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                if first_token:
                    ttft_ms = int((time.perf_counter() - req_start) * 1000)
                    push({"type": "ttft", "ttft_ms": ttft_ms})
                    logger.info("stream TTFT=%dms", ttft_ms)
                    first_token = False
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    push({"type": "chunk", "data": json.loads(data)})
                except json.JSONDecodeError:
                    push({"type": "error", "message": data})
                    return
        push({"type": "done"})
    except httpx.HTTPStatusError as exc:
        push({
            "type": "error",
            "status_code": exc.response.status_code,
            "message": exc.response.text[:2000],
        })
    except Exception as exc:
        push({"type": "error", "message": str(exc)})


def history_messages(history: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in history or []:
        role = item.get("role")
        content = history_item_content(item)
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


def history_item_content(item: dict[str, Any]) -> Any:
    extra = {key: value for key, value in item.items() if key != "role"}
    if len(extra) > 1:
        return json.dumps(extra, ensure_ascii=False)

    content = item.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def apply_provider_options(
    payload: dict[str, Any],
    provider: Optional[str] = None,
    modalities: Optional[list[str]] = None,
) -> dict[str, Any]:
    if (provider or resolved_provider()) == "vllm_omni":
        payload["modalities"] = modalities or QWEN_MODALITIES
        if QWEN_SPEAKER:
            payload["speaker"] = QWEN_SPEAKER
    return payload


def json_object_response_format() -> dict[str, str]:
    return {"type": "json_object"}


def build_chat_payload(
    model: str,
    wav_bytes: bytes,
    prompt: str,
    *,
    history: Optional[list[dict[str, Any]]] = None,
    max_tokens: int,
    modalities: Optional[list[str]] = None,
    response_format: Optional[dict[str, Any]] = None,
    turn_instruction: str | None = None,
) -> dict[str, Any]:
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    provider = resolved_provider()
    if provider == "vllm_omni":
        audio_item = {
            "type": "audio_url",
            "audio_url": {
                "url": f"data:audio/wav;base64,{audio_b64}",
            },
        }
    else:
        audio_item = {
            "type": "audio",
            "audio": f"data:audio/wav;base64,{audio_b64}",
        }

    messages: list[dict[str, Any]] = []
    prompt = prompt.strip()
    if prompt:
        messages.append({"role": "system", "content": prompt})
    messages.extend(history_messages(history))
    instruction = (turn_instruction or "请理解这段语音输入，并直接回答用户。").strip()
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": instruction,
                },
                audio_item,
            ],
        }
    )

    payload = {
        "model": model,
        "messages": messages,
        "chat_template_kwargs": {"sampling_rate": TARGET_SAMPLE_RATE},
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if response_format:
        payload["response_format"] = response_format
    return apply_provider_options(payload, provider, modalities=modalities or ["text"])


def build_text_payload(
    model: str,
    user_text: str,
    prompt: str,
    *,
    history: Optional[list[dict[str, Any]]] = None,
    max_tokens: int,
    modalities: Optional[list[str]] = None,
    response_format: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    messages = history_messages(history)
    content = user_text.strip()
    prompt = prompt.strip()
    if prompt and content:
        content = f"{prompt}\n\n用户输入：{content}"
    elif prompt:
        content = prompt

    provider = resolved_provider()
    payload = {
        "model": model,
        "messages": [
            *messages,
            {
                "role": "user",
                "content": content,
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if provider == "vllm_omni":
        payload["chat_template_kwargs"] = {"sampling_rate": TARGET_SAMPLE_RATE}
    if response_format:
        payload["response_format"] = response_format
    return apply_provider_options(payload, provider, modalities=modalities or ["text"])


def normalize_history(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    items: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = history_item_content(item)
        if not content:
            continue
        if isinstance(content, str):
            content = content[:4000]
        items.append({"role": role, "content": content})

    max_messages = max(0, MAX_HISTORY_TURNS * 2)
    if max_messages:
        return items[-max_messages:]
    return []


def extract_model_output(data: dict[str, Any]) -> dict[str, Optional[str]]:
    text_parts: list[str] = []
    audio_data_url: Optional[str] = None

    for choice in data.get("choices") or []:
        message = choice.get("message") or {}
        content = message.get("content")

        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text"):
                    text_parts.append(str(item["text"]))
                if audio_data_url is None:
                    audio_data_url = find_audio_data_url(item)

        openai_audio = message.get("audio")
        if isinstance(openai_audio, dict):
            transcript = openai_audio.get("transcript")
            if transcript:
                text_parts.append(str(transcript))
            if audio_data_url is None and openai_audio.get("data"):
                audio_data_url = as_audio_data_url(str(openai_audio["data"]))

        if audio_data_url is None:
            audio_data_url = find_audio_data_url(message)

    if audio_data_url is None:
        audio_data_url = find_audio_data_url(data)

    return {
        "text": "\n".join(part for part in text_parts if part).strip() or None,
        "audio_data_url": audio_data_url,
    }


def extract_stream_delta(data: dict[str, Any]) -> dict[str, Optional[str]]:
    text: Optional[str] = None
    audio_data_url: Optional[str] = None
    modality = data.get("modality")

    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if modality == "text" and isinstance(content, str):
            text = (text or "") + content
        elif modality == "audio" and isinstance(content, str):
            audio_data_url = as_audio_data_url(content)
        elif isinstance(content, str):
            if content.startswith("data:audio/") or len(content) > 200:
                audio_data_url = as_audio_data_url(content)
            else:
                text = (text or "") + content
        if audio_data_url is None:
            audio_data_url = find_audio_data_url(delta)

    if audio_data_url is None:
        audio_data_url = find_audio_data_url(data)
    return {"text": text, "audio_data_url": audio_data_url}


def as_audio_data_url(value: str) -> str:
    if value.startswith("data:audio/"):
        return value
    return "data:audio/wav;base64," + value


def find_audio_data_url(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        if obj.startswith("data:audio/"):
            return obj
        return None
    if isinstance(obj, list):
        for item in obj:
            found = find_audio_data_url(item)
            if found:
                return found
        return None
    if not isinstance(obj, dict):
        return None

    for key in ("audio", "audio_url", "output_audio", "audio_data"):
        value = obj.get(key)
        if isinstance(value, str):
            if value.startswith("data:audio/"):
                return value
            if len(value) > 200:
                return as_audio_data_url(value)
        if isinstance(value, dict):
            audio_value = value.get("url") or value.get("data") or value.get("base64")
            if isinstance(audio_value, str):
                return as_audio_data_url(audio_value)

    for value in obj.values():
        found = find_audio_data_url(value)
        if found:
            return found
    return None
