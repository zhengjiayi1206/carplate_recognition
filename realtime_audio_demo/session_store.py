import asyncio
import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from realtime_audio_demo.config import MAX_HISTORY_TURNS, SESSION_TTL
from realtime_audio_demo.services.plate_agent import PlateAgentState, clone_state

logger = logging.getLogger("uvicorn.error")


@dataclass
class ChatSession:
    session_id: str
    history: list[dict[str, Any]] = field(default_factory=list)
    plate_state: PlateAgentState = field(default_factory=PlateAgentState)
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)


_sessions: dict[str, ChatSession] = {}
_lock = asyncio.Lock()


async def get_session(session_id: str) -> ChatSession:
    async with _lock:
        session = _sessions.get(session_id)
        if session is None:
            session = ChatSession(session_id=session_id)
            _sessions[session_id] = session
            logger.info("session %s created", session_id)
        session.last_access = time.time()
        return session


async def append_history(session_id: str, role: str, content: Any) -> None:
    if isinstance(content, str) and not content.strip():
        return
    if content is None:
        return
    async with _lock:
        session = _sessions.get(session_id)
        if session is None:
            session = ChatSession(session_id=session_id)
            _sessions[session_id] = session
            logger.info("session %s created", session_id)
        stored_content = content[:4000] if isinstance(content, str) else content
        session.history.append({"role": role, "content": stored_content})
        max_messages = max(0, MAX_HISTORY_TURNS * 2)
        if max_messages and len(session.history) > max_messages:
            session.history = session.history[-max_messages:]
        session.last_access = time.time()


async def append_audio_history(session_id: str, wav_bytes: bytes) -> None:
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    await append_history(
        session_id,
        "user",
        [
            {
                "type": "audio_url",
                "audio_url": {
                    "url": f"data:audio/wav;base64,{audio_b64}",
                },
            },
            {
                "type": "text",
                "text": "这是用户上一轮语音输入。",
            },
        ],
    )


async def get_session_history(session_id: str) -> list[dict[str, Any]]:
    async with _lock:
        session = _sessions.get(session_id)
        if session is None:
            session = ChatSession(session_id=session_id)
            _sessions[session_id] = session
            logger.info("session %s created", session_id)
        session.last_access = time.time()
        return list(session.history)


async def get_plate_agent_state(session_id: str) -> PlateAgentState:
    async with _lock:
        session = _sessions.get(session_id)
        if session is None:
            session = ChatSession(session_id=session_id)
            _sessions[session_id] = session
            logger.info("session %s created", session_id)
        session.last_access = time.time()
        return clone_state(session.plate_state)


async def update_plate_agent_state(session_id: str, state: PlateAgentState) -> None:
    async with _lock:
        session = _sessions.get(session_id)
        if session is None:
            session = ChatSession(session_id=session_id)
            _sessions[session_id] = session
            logger.info("session %s created", session_id)
        session.plate_state = clone_state(state)
        session.last_access = time.time()


async def reset_session_state(session_id: str) -> None:
    async with _lock:
        session = _sessions.get(session_id)
        if session is None:
            session = ChatSession(session_id=session_id)
            _sessions[session_id] = session
            logger.info("session %s created", session_id)
        session.history = []
        session.plate_state = PlateAgentState()
        session.last_access = time.time()


async def delete_session(session_id: str) -> None:
    async with _lock:
        if session_id in _sessions:
            del _sessions[session_id]
            logger.info("session %s deleted", session_id)


async def cleanup_expired_sessions() -> int:
    now = time.time()
    async with _lock:
        expired = [
            sid
            for sid, s in _sessions.items()
            if now - s.last_access > SESSION_TTL
        ]
        for sid in expired:
            del _sessions[sid]
    if expired:
        logger.info("cleaned up %d expired sessions", len(expired))
    return len(expired)
