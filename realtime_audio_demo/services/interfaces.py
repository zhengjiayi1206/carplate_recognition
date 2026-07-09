from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional


@dataclass(slots=True)
class CompletionResult:
    text: str | None
    audio_data_url: str | None
    raw_response: dict[str, Any] | None
    latency_ms: int
    ttft_ms: int | None = None


@dataclass(slots=True)
class StreamItem:
    type: str
    data: dict[str, Any] | None = None
    text: str | None = None
    audio_data_url: str | None = None
    status_code: int | None = None
    message: str | None = None
    ttft_ms: int | None = None


@dataclass(slots=True)
class TurnDecision:
    turn_state: str
    transcription: str
    raw_output: str
    latency_ms: int
    audio_seconds: float

    @property
    def is_incomplete(self) -> bool:
        return self.turn_state == "INCOMPLETE"


class ChatModel(ABC):
    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError


class SpeechSynthesizer(ABC):
    @abstractmethod
    async def synthesize(self, *, model: str, text: str) -> str | None:
        raise NotImplementedError


class TurnDetector(ABC):
    @abstractmethod
    async def judge(self, *, chunks: list[bytes], sample_rate: int) -> TurnDecision:
        raise NotImplementedError


class AudioTranscoder(ABC):
    @abstractmethod
    def float32_chunks_to_wav(self, chunks: list[bytes], source_rate: int, target_rate: int) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def save_wav(self, wav_bytes: bytes, output_path: Path) -> None:
        raise NotImplementedError
