import base64
import logging
from typing import Any, Callable

import httpx

from realtime_audio_demo.config import (
    FINAL_MAX_TOKENS,
    TTS_API_BASE,
    TTS_MODEL,
    TTS_REF_AUDIO,
    TTS_REF_TEXT,
    TTS_RESPONSE_FORMAT,
    TTS_STREAM_FORMAT,
    TTS_STREAM_RESPONSE_FORMAT,
    TTS_TASK_TYPE,
    TTS_VOICE,
)
from realtime_audio_demo.services.interfaces import ChatModel, SpeechSynthesizer
from realtime_audio_demo.services.model_gateway import model_gateway

logger = logging.getLogger("uvicorn.error")


def build_tts_payload(*, model: str, text: str, response_format: str, stream: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": model,
        "input": text,
        "response_format": response_format,
    }
    if TTS_VOICE:
        payload["voice"] = TTS_VOICE
    if TTS_TASK_TYPE:
        payload["task_type"] = TTS_TASK_TYPE
    if TTS_REF_AUDIO:
        payload["ref_audio"] = TTS_REF_AUDIO
        payload.setdefault("task_type", "Base")
    if TTS_REF_TEXT:
        payload["ref_text"] = TTS_REF_TEXT
    if stream:
        payload["stream"] = True
        payload["stream_format"] = TTS_STREAM_FORMAT
    return payload


class TtsApiSpeechSynthesizer(SpeechSynthesizer):
    """使用外部 TTS API（OpenAI /v1/audio/speech 格式）合成语音。"""

    def __init__(self, api_base: str) -> None:
        self.api_base = api_base.rstrip("/")

    async def synthesize(self, *, model: str, text: str) -> str | None:
        speech_text = text.strip()
        if not speech_text:
            return None
        if not self.api_base:
            logger.warning("TTS_API_BASE not configured, skipping TTS")
            return None

        audio_data_url = await self._request_tts(model=TTS_MODEL, text=speech_text)
        return audio_data_url

    async def _request_tts(self, *, model: str, text: str) -> str | None:
        url = f"{self.api_base}/v1/audio/speech"
        payload = build_tts_payload(model=model, text=text, response_format=TTS_RESPONSE_FORMAT)

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
                response = await client.post(url, json=payload)
        except Exception as exc:
            logger.error("TTS request failed: %s", exc)
            return None

        if response.status_code >= 400:
            logger.error("TTS error: status=%d body=%s", response.status_code, response.text[:500])
            return None

        wav_bytes = response.content
        if not wav_bytes:
            logger.warning("TTS returned empty body")
            return None

        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        return f"data:audio/{TTS_RESPONSE_FORMAT};base64,{audio_b64}"

    async def stream_synthesize(
        self,
        *,
        model: str,
        text: str,
        on_audio_chunk: Callable[[bytes], Any],
    ) -> bool:
        speech_text = text.strip()
        if not speech_text:
            return False
        if not self.api_base:
            logger.warning("TTS_API_BASE not configured, skipping streaming TTS")
            return False

        url = f"{self.api_base}/v1/audio/speech"
        payload = build_tts_payload(
            model=TTS_MODEL,
            text=speech_text,
            response_format=TTS_STREAM_RESPONSE_FORMAT,
            stream=True,
        )
        sent = False
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        logger.error("streaming TTS error: status=%d body=%s", response.status_code, body[:500])
                        return False
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        await on_audio_chunk(chunk)
                        sent = True
        except Exception as exc:
            logger.error("streaming TTS request failed: %s", exc)
            return False
        return sent


class ModelSpeechSynthesizer(SpeechSynthesizer):
    """（旧）使用 Qwen 模型本身合成语音——通过 chat completion + output_audio。"""

    def __init__(self, model_client: ChatModel) -> None:
        self.model_client = model_client

    async def synthesize(self, *, model: str, text: str) -> str | None:
        speech_text = text.strip()
        if not speech_text:
            return None

        result, status_code = await self.model_client.complete_text(
            model=model,
            text=speech_text,
            prompt="请把用户输入作为语音播报文本。只按原文朗读，不要解释、不要改写、不要补充任何内容。",
            history=[],
            max_tokens=FINAL_MAX_TOKENS,
            output_audio=True,
        )
        if status_code >= 400:
            return None
        audio_data_url = result.get("audio_data_url")
        return str(audio_data_url) if audio_data_url else None

    async def stream_synthesize(
        self,
        *,
        model: str,
        text: str,
        on_audio_chunk: Callable[[bytes], Any],
    ) -> bool:
        return False


def create_speech_synthesizer() -> SpeechSynthesizer:
    """优先使用外部 TTS API（TTS_API_BASE），否则退回到模型自带 TTS。"""
    tts_base = TTS_API_BASE.strip()
    if tts_base:
        logger.info("using external TTS API: %s", tts_base)
        return TtsApiSpeechSynthesizer(tts_base)
    logger.info("TTS_API_BASE not set, falling back to model-based TTS")
    return ModelSpeechSynthesizer(model_gateway)


speech_synthesizer = create_speech_synthesizer()
