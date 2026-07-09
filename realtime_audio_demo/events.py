import json
from typing import Any

from realtime_audio_demo.sessions import AudioSession


async def send_event(session: AudioSession, event_type: str, payload: dict[str, Any]) -> None:
    await session.websocket.send_text(json.dumps({"type": event_type, **payload}, ensure_ascii=False))

