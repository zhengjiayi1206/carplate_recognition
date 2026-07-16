import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import WebSocket

from realtime_audio_demo.config import DEFAULT_PREFILL_MS, QWEN_MODEL, TARGET_SAMPLE_RATE


@dataclass
class AudioSession:
    websocket: WebSocket
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    chat_session_id: Optional[str] = None
    source_sample_rate: int = 48000
    target_sample_rate: int = TARGET_SAMPLE_RATE
    model: str = QWEN_MODEL
    prompt: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    route_context: str = ""
    conversation_mode: str = "manual"
    easy_turn_enabled: bool = False
    output_audio: bool = False
    stream_speech_audio: bool = False
    prefill_ms: int = DEFAULT_PREFILL_MS
    chunks: list[bytes] = field(default_factory=list)
    prefill_queue: asyncio.Queue[tuple[int, bytes]] = field(default_factory=asyncio.Queue)
    prefill_task: Optional[asyncio.Task] = None
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    vad: Any = None
    started_at: float = field(default_factory=time.time)
    stopped: bool = False
