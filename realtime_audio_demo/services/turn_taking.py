import asyncio
import base64
import logging
import time

import httpx

from realtime_audio_demo.config import EASYTURN_API_URL, REQUEST_TIMEOUT, TARGET_SAMPLE_RATE
from realtime_audio_demo.services.interfaces import TurnDecision, TurnDetector
from realtime_audio_demo.utils.audio import wav_bytes_from_float32_chunks


logger = logging.getLogger("uvicorn.error")


class EasyTurnDetector(TurnDetector):
    async def judge(self, *, chunks: list[bytes], sample_rate: int) -> TurnDecision:
        if not EASYTURN_API_URL:
            raise RuntimeError("EASYTURN_API_URL is required when EASYTURN_ENABLED=1")
        return await asyncio.to_thread(self._judge_sync, list(chunks), sample_rate)

    def _judge_sync(self, chunks: list[bytes], sample_rate: int) -> TurnDecision:
        start = time.perf_counter()
        wav_bytes = wav_bytes_from_float32_chunks(chunks, sample_rate, TARGET_SAMPLE_RATE)
        payload = {
            "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
            "audio_format": "wav",
            "source_sample_rate": sample_rate,
            "target_sample_rate": TARGET_SAMPLE_RATE,
        }
        with httpx.Client(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0)) as client:
            response = client.post(EASYTURN_API_URL, json=payload)
            response.raise_for_status()
            data = response.json()

        latency_ms = int((time.perf_counter() - start) * 1000)
        turn_state = str(data.get("turn_state") or data.get("state") or "").strip().upper()
        if not turn_state:
            raise RuntimeError("EasyTurn API response missing turn_state")

        return TurnDecision(
            turn_state=turn_state,
            transcription=str(data.get("transcription") or data.get("text") or ""),
            raw_output=str(data.get("raw_output") or data.get("raw") or data),
            latency_ms=int(data.get("latency_ms") or latency_ms),
            audio_seconds=float(data.get("audio_seconds") or 0.0),
        )


easy_turn_detector = EasyTurnDetector()
