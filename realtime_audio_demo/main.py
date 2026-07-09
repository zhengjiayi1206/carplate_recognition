import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from realtime_audio_demo.config import (
    EASYTURN_API_URL,
    EASYTURN_ENABLED,
    SILERO_VAD_ENABLED,
    SILERO_VAD_PRELOAD,
    STATIC_DIR,
)
from realtime_audio_demo.routes import audio, pages
from realtime_audio_demo.services.silero_vad import SileroVadUnavailable, preload_silero_vad
from realtime_audio_demo.session_store import cleanup_expired_sessions


logger = logging.getLogger(__name__)


async def _session_cleanup_loop() -> None:
    while True:
        try:
            await asyncio.sleep(300)
        except asyncio.CancelledError:
            return
        try:
            await cleanup_expired_sessions()
        except Exception:
            logger.exception("session cleanup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if SILERO_VAD_ENABLED and SILERO_VAD_PRELOAD:
        try:
            status = await asyncio.to_thread(preload_silero_vad)
            app.state.silero_vad = status
            logger.info("Silero VAD preloaded on %s", status.get("device"))
        except SileroVadUnavailable as exc:
            app.state.silero_vad = {"preloaded": False, "error": str(exc)}
            logger.warning("Silero VAD preload skipped: %s", exc)
        except Exception as exc:
            app.state.silero_vad = {"preloaded": False, "error": str(exc)}
            logger.exception("Silero VAD preload failed")

    logger.info("EasyTurn enabled=%s api_configured=%s", EASYTURN_ENABLED, bool(EASYTURN_API_URL))

    cleanup_task = asyncio.create_task(_session_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="Carplate chatbox realtime demo",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(pages.router)
    app.include_router(audio.router)
    return app


app = create_app()
