from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from realtime_audio_demo.config import (
    EASYTURN_ACK_TEXT,
    EASYTURN_API_URL,
    EASYTURN_ENABLED,
    MAX_HISTORY_TURNS,
    PREFILL_MODE,
    QWEN_API_BASE,
    QWEN_MODALITIES,
    QWEN_MODEL,
    QWEN_SPEAKER,
    SESSION_TTL,
    SILERO_VAD_ENABLED,
    SILERO_VAD_MAX_SPEECH_MS,
    SILERO_VAD_MIN_SILENCE_MS,
    SILERO_VAD_MIN_SPEECH_MS,
    SILERO_VAD_PRELOAD,
    SILERO_VAD_THRESHOLD,
    STATIC_DIR,
    STREAM_FINAL_OUTPUT,
    SYSTEM_PROMPT_PATH,
    TARGET_SAMPLE_RATE,
    resolved_provider,
)
from realtime_audio_demo.services.silero_vad import silero_vad_status


router = APIRouter()
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, max-age=0",
    "Pragma": "no-cache",
}


@router.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse("/chatbox")


@router.get("/chatbox")
async def chatbox() -> FileResponse:
    return FileResponse(STATIC_DIR / "chatbox.html", headers=NO_CACHE_HEADERS)


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "qwen_api_base": QWEN_API_BASE,
            "model": QWEN_MODEL,
            "prefill_mode": PREFILL_MODE,
            "provider": resolved_provider(),
            "modalities": QWEN_MODALITIES,
            "speaker": QWEN_SPEAKER or None,
            "target_sample_rate": TARGET_SAMPLE_RATE,
            "max_history_turns": MAX_HISTORY_TURNS,
            "stream_final_output": STREAM_FINAL_OUTPUT,
            "silero_vad": {
                "enabled": SILERO_VAD_ENABLED,
                "preload": SILERO_VAD_PRELOAD,
                "threshold": SILERO_VAD_THRESHOLD,
                "min_speech_ms": SILERO_VAD_MIN_SPEECH_MS,
                "min_silence_ms": SILERO_VAD_MIN_SILENCE_MS,
                "max_speech_ms": SILERO_VAD_MAX_SPEECH_MS,
                "status": silero_vad_status(),
                "startup": getattr(request.app.state, "silero_vad", None),
            },
            "easy_turn": {
                "enabled": EASYTURN_ENABLED,
                "api_configured": bool(EASYTURN_API_URL),
                "ack_text": EASYTURN_ACK_TEXT,
            },
            "system_prompt_path": str(SYSTEM_PROMPT_PATH),
            "session_ttl": SESSION_TTL,
        }
    )
