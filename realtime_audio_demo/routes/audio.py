import asyncio
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from realtime_audio_demo.config import (
    CAPTURE_DIR,
    DEFAULT_PREFILL_MS,
    EASYTURN_ACK_TEXT,
    EASYTURN_ENABLED,
    FINAL_MAX_TOKENS,
    PREFILL_MODE,
    QWEN_API_BASE,
    QWEN_MODEL,
    SILERO_VAD_ENABLED,
    TTS_SAMPLE_RATE,
    TTS_STREAM_RESPONSE_FORMAT,
    normalize_model_name,
)
from realtime_audio_demo.events import send_event
from realtime_audio_demo.routes.request_utils import read_json_object
from realtime_audio_demo.services.chatbox_application import chatbox_service
from realtime_audio_demo.services.qwen import (
    build_chat_payload,
    normalize_history,
    post_json,
)
from realtime_audio_demo.services.audio_processing import audio_transcoder
from realtime_audio_demo.services.model_gateway import model_gateway
from realtime_audio_demo.services.plate_agent import PlateAgentState, get_plate_agent_service
from realtime_audio_demo.services.prompt_provider import system_prompt_provider
from realtime_audio_demo.services.speech import speech_synthesizer
from realtime_audio_demo.services.silero_vad import SileroVadConfig, SileroVadSession, SileroVadUnavailable
from realtime_audio_demo.services.turn_taking import easy_turn_detector
from realtime_audio_demo.session_store import (
    append_audio_history,
    append_plate_audio_turn,
    append_history,
    build_plate_audio_input,
    clear_plate_audio_turns,
    get_plate_agent_state,
    get_session_history,
    update_plate_agent_state,
)
from realtime_audio_demo.sessions import AudioSession
from realtime_audio_demo.utils.audio import float32_sample_count, wav_bytes_from_float32_chunks


router = APIRouter()
logger = logging.getLogger("uvicorn.error")


def sse_data_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_speech_sse_events(
    *,
    model: str,
    text: str,
    segment: str,
    audio_index: int,
):
    speech_text = text.strip()
    if not speech_text:
        return
    yield sse_data_event(
        {
            "stage": "tts_audio_start",
            "segment": segment,
            "audio_index": audio_index,
            "format": TTS_STREAM_RESPONSE_FORMAT,
            "sample_rate": TTS_SAMPLE_RATE,
            "speech_text": speech_text,
        }
    )

    chunk_queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def on_chunk(chunk: bytes) -> None:
        await chunk_queue.put(chunk)

    tts_task = asyncio.create_task(
        speech_synthesizer.stream_synthesize(
            model=model,
            text=speech_text,
            on_audio_chunk=on_chunk,
        )
    )
    while not tts_task.done() or not chunk_queue.empty():
        try:
            chunk = await asyncio.wait_for(chunk_queue.get(), timeout=0.2)
        except asyncio.TimeoutError:
            continue
        yield sse_data_event(
            {
                "stage": "tts_audio_delta",
                "segment": segment,
                "audio_index": audio_index,
                "format": TTS_STREAM_RESPONSE_FORMAT,
                "sample_rate": TTS_SAMPLE_RATE,
                "audio_base64": base64.b64encode(chunk).decode("ascii"),
            }
        )
        chunk_queue.task_done()

    ok = await tts_task
    if ok:
        yield sse_data_event(
            {
                "stage": "tts_audio_end",
                "segment": segment,
                "audio_index": audio_index,
                "format": TTS_STREAM_RESPONSE_FORMAT,
                "sample_rate": TTS_SAMPLE_RATE,
                "speech_text": speech_text,
            }
        )
    else:
        yield sse_data_event(
            {
                "stage": "tts_audio_error",
                "segment": segment,
                "audio_index": audio_index,
                "message": "streaming TTS failed or returned no audio",
            }
        )


@router.post("/api/chatbox/text")
async def chatbox_text(request: Request) -> JSONResponse:
    payload, error_response = await read_json_object(request)
    if error_response is not None:
        return error_response
    result, status_code = await chatbox_service.handle_text(payload)
    return JSONResponse(result, status_code=status_code)


@router.post("/api/chatbox/session/start")
async def chatbox_session_start(request: Request) -> JSONResponse:
    payload, error_response = await read_json_object(request)
    if error_response is not None:
        return error_response
    result, status_code = await chatbox_service.start_session(payload)
    return JSONResponse(result, status_code=status_code)


@router.post("/api/chatbox/speech")
async def chatbox_speech(request: Request) -> JSONResponse:
    payload, error_response = await read_json_object(request)
    if error_response is not None:
        return error_response

    return JSONResponse(await chatbox_service.synthesize_speech(payload))


@router.post("/api/chatbox/audio")
async def chatbox_audio(request: Request) -> JSONResponse:
    payload, error_response = await read_json_object(request)
    if error_response is not None:
        return error_response
    result, status_code = await chatbox_service.handle_audio(payload)
    return JSONResponse(result, status_code=status_code)


@router.post("/api/chatbox/audio/stream")
async def chatbox_audio_stream(request: Request) -> StreamingResponse:
    payload, error_response = await read_json_object(request)
    if error_response is not None:
        return error_response

    audio_base64 = str(payload.get("audio_base64") or "").strip()
    if not audio_base64:
        return JSONResponse({"message": "audio_base64 is required"}, status_code=400)
    if "," in audio_base64 and audio_base64.startswith("data:"):
        audio_base64 = audio_base64.split(",", 1)[1]
    try:
        wav_bytes = base64.b64decode(audio_base64, validate=True)
    except Exception as exc:
        return JSONResponse({"message": f"bad audio_base64: {exc}"}, status_code=400)

    model = normalize_model_name(payload.get("model") or QWEN_MODEL)
    session_id = str(payload.get("session_id") or "").strip()
    state = await get_plate_agent_state(session_id) if session_id else PlateAgentState()
    turn_wav_bytes = wav_bytes
    input_wav_bytes = wav_bytes
    if session_id:
        input_wav_bytes = await build_plate_audio_input(
            session_id,
            turn_wav_bytes,
            has_plate=state.has_car_plate,
        )
    agent = get_plate_agent_service(model_gateway)
    should_synthesize = bool(payload.get("outputAudio"))

    async def event_stream():
        ack_queue: asyncio.Queue[str] = asyncio.Queue()
        audio_index = 0

        async def on_ack(text: str) -> None:
            await ack_queue.put(text)

        agent_task = asyncio.create_task(
            agent.handle_audio_turn(
                model=model, wav_bytes=input_wav_bytes, state=state, on_ack=on_ack,
            )
        )

        # Yield ack events as they arrive from the agent
        while not agent_task.done() or not ack_queue.empty():
            try:
                ack_text = await asyncio.wait_for(ack_queue.get(), timeout=0.2)
                ack_event: dict[str, Any] = {"stage": "ack", "speech_text": ack_text}
                yield sse_data_event(ack_event)
                if should_synthesize:
                    audio_index += 1
                    async for tts_event in stream_speech_sse_events(
                        model=model,
                        text=ack_text,
                        segment="ack",
                        audio_index=audio_index,
                    ):
                        yield tts_event
            except asyncio.TimeoutError:
                continue

        agent_result = await agent_task

        # Persist state
        if session_id:
            await append_audio_history(session_id, turn_wav_bytes)
            await append_history(session_id, "assistant", agent_result.history_text)
            await update_plate_agent_state(session_id, agent_result.state)
            if not state.has_car_plate:
                if agent_result.state.has_car_plate:
                    await clear_plate_audio_turns(session_id)
                else:
                    await append_plate_audio_turn(session_id, turn_wav_bytes)

        result_event = {
            "stage": "result",
            "text": agent_result.text,
            "audio_data_url": None,
            "history_text": agent_result.history_text,
            "speech_text": agent_result.speech_text,
            "output_is_json": True,
            "latency_ms": agent_result.latency_ms,
            "agent_state": agent_result.state.to_context(),
        }
        yield sse_data_event(result_event)
        if should_synthesize:
            audio_index += 1
            async for tts_event in stream_speech_sse_events(
                model=model,
                text=agent_result.speech_text,
                segment="final",
                audio_index=audio_index,
            ):
                yield tts_event
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket) -> None:
    await websocket.accept()
    session = AudioSession(websocket=websocket)
    session.prefill_task = asyncio.create_task(prefill_worker(session))
    await send_event(session, "ready", {"session_id": session.session_id})

    try:
        while True:
            try:
                message = await websocket.receive()
            except RuntimeError as exc:
                if "disconnect message has been received" in str(exc):
                    break
                raise
            if message.get("text") is not None:
                await handle_control_message(session, message["text"])
            elif message.get("bytes") is not None:
                await handle_audio_chunk(session, message["bytes"])
    except WebSocketDisconnect:
        session.stopped = True
    finally:
        if session.prefill_task:
            session.prefill_task.cancel()
            try:
                await session.prefill_task
            except asyncio.CancelledError:
                pass


@router.websocket("/ws/vad")
async def websocket_vad(websocket: WebSocket) -> None:
    await websocket.accept()
    session = AudioSession(websocket=websocket)
    await send_event(session, "ready", {"session_id": session.session_id})

    try:
        while True:
            try:
                message = await websocket.receive()
            except RuntimeError as exc:
                if "disconnect message has been received" in str(exc):
                    break
                raise
            if message.get("text") is not None:
                await handle_vad_control_message(session, message["text"])
            elif message.get("bytes") is not None:
                await handle_vad_monitor_chunk(session, message["bytes"])
    except WebSocketDisconnect:
        session.stopped = True


async def handle_vad_control_message(session: AudioSession, text: str) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        await send_event(session, "error", {"message": f"bad json: {exc}"})
        return

    event_type = payload.get("type")
    if event_type == "start":
        session.source_sample_rate = int(payload.get("sampleRate") or session.source_sample_rate)
        await configure_vad(session, payload.get("vad") or payload)
        await send_event(
            session,
            "vad_monitor_started",
            {
                "session_id": session.session_id,
                "source_sample_rate": session.source_sample_rate,
                "vad": "silero" if session.vad else "none",
            },
        )
    elif event_type == "stop":
        session.stopped = True
        await send_event(session, "vad_monitor_stopped", {"session_id": session.session_id})
    elif event_type == "reset_vad":
        if session.vad is not None:
            session.vad.reset()
        await send_event(session, "vad_reset", {"session_id": session.session_id})
    elif event_type == "ping":
        await send_event(session, "pong", {"ts": time.time()})
    else:
        await send_event(session, "error", {"message": f"unknown control event: {event_type}"})


async def handle_control_message(session: AudioSession, text: str) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        await send_event(session, "error", {"message": f"bad json: {exc}"})
        return

    event_type = payload.get("type")
    if event_type == "start":
        session.source_sample_rate = int(payload.get("sampleRate") or session.source_sample_rate)
        session.prefill_ms = int(payload.get("prefillMs") or DEFAULT_PREFILL_MS)
        session.model = normalize_model_name(payload.get("model") or QWEN_MODEL)
        session.route_context = str(payload.get("routeContext") or "").strip()
        session.conversation_mode = normalize_conversation_mode(payload.get("conversationMode"))
        session.easy_turn_enabled = bool(payload.get("easyTurnEnabled")) and EASYTURN_ENABLED
        session.output_audio = bool(payload.get("outputAudio"))
        session.stream_speech_audio = bool(payload.get("streamSpeechAudio"))
        await configure_vad(session, payload.get("vad"))

        # Load session_id from client
        chat_session_id = str(payload.get("session_id") or "").strip()
        session.chat_session_id = chat_session_id or None

        session.prompt = system_prompt_provider.get_prompt()

        # Handle barge-in: append interrupted assistant text to session history
        if chat_session_id:
            interrupted_text = str(payload.get("interrupted_assistant_text") or "").strip()
            if interrupted_text:
                await append_history(chat_session_id, "assistant", interrupted_text)

        if chat_session_id:
            session.history = normalize_history(await get_session_history(chat_session_id))
        else:
            session.history = normalize_history(payload.get("history"))
        await send_event(
            session,
            "started",
            {
                "session_id": session.session_id,
                "source_sample_rate": session.source_sample_rate,
                "target_sample_rate": session.target_sample_rate,
                "prefill_ms": session.prefill_ms,
                "model": session.model,
                "qwen_api_base": QWEN_API_BASE,
                "conversation_mode": session.conversation_mode,
                "easy_turn_enabled": session.easy_turn_enabled,
                "history_messages": len(session.history),
                "output_audio": session.output_audio,
                "stream_speech_audio": session.stream_speech_audio,
                "vad": "silero" if session.vad else "none",
            },
        )
    elif event_type == "stop":
        logger.info(
            "audio stop received session=%s mode=%s chunks=%d easyturn=%s",
            session.session_id,
            session.conversation_mode,
            len(session.chunks),
            session.easy_turn_enabled,
        )
        await send_event(
            session,
            "finalizing",
            {"chunks": len(session.chunks), "easy_turn_enabled": session.easy_turn_enabled},
        )
        if session.conversation_mode == "realtime" and session.easy_turn_enabled and await maybe_handle_easy_turn_stop(session):
            return
        session.stopped = True
        await finalize_session(session)
    elif event_type == "ping":
        await send_event(session, "pong", {"ts": time.time()})
    else:
        await send_event(session, "error", {"message": f"unknown control event: {event_type}"})


def normalize_conversation_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode == "realtime":
        return "realtime"
    return "manual"


async def synthesize_speech_audio(model: str, text: str | None) -> str | None:
    speech_text = (text or "").strip()
    if not speech_text:
        return None
    return await speech_synthesizer.synthesize(model=model, text=speech_text)


async def stream_speech_audio_events(
    session: AudioSession,
    *,
    text: str,
    segment: str,
    audio_index: int,
) -> bool:
    speech_text = text.strip()
    if not speech_text:
        return False

    await send_event(
        session,
        "tts_audio_start",
        {
            "segment": segment,
            "audio_index": audio_index,
            "format": TTS_STREAM_RESPONSE_FORMAT,
            "sample_rate": TTS_SAMPLE_RATE,
            "speech_text": speech_text,
        },
    )

    async def on_chunk(chunk: bytes) -> None:
        await send_event(
            session,
            "tts_audio_delta",
            {
                "segment": segment,
                "audio_index": audio_index,
                "format": TTS_STREAM_RESPONSE_FORMAT,
                "sample_rate": TTS_SAMPLE_RATE,
                "audio_base64": base64.b64encode(chunk).decode("ascii"),
            },
        )

    ok = await speech_synthesizer.stream_synthesize(
        model=session.model,
        text=speech_text,
        on_audio_chunk=on_chunk,
    )
    if ok:
        await send_event(
            session,
            "tts_audio_end",
            {
                "segment": segment,
                "audio_index": audio_index,
                "format": TTS_STREAM_RESPONSE_FORMAT,
                "sample_rate": TTS_SAMPLE_RATE,
                "speech_text": speech_text,
            },
        )
    else:
        await send_event(
            session,
            "tts_audio_error",
            {
                "segment": segment,
                "audio_index": audio_index,
                "message": "streaming TTS failed or returned no audio",
            },
        )
    return ok


async def maybe_handle_easy_turn_stop(session: AudioSession) -> bool:
    if not EASYTURN_ENABLED:
        return False
    if not session.chunks:
        logger.info("easyturn skipped: no audio chunks session=%s", session.session_id)
        return False

    logger.info("easyturn judging session=%s chunks=%d source_sr=%d", session.session_id, len(session.chunks), session.source_sample_rate)
    try:
        decision = await easy_turn_detector.judge(chunks=list(session.chunks), sample_rate=session.source_sample_rate)
    except Exception as exc:
        logger.exception("EasyTurn judge failed; fallback to final inference: %s", exc)
        await send_event(session, "turn_error", {"message": str(exc)})
        return False

    logger.info(
        "easyturn state=%s latency=%dms audio=%.2fs raw=%r",
        decision.turn_state,
        decision.latency_ms,
        decision.audio_seconds,
        decision.raw_output,
    )

    if decision.turn_state != "INCOMPLETE":
        logger.info("easyturn final route: state=%s -> qwen final inference", decision.turn_state)
        await send_event(
            session,
            "turn_complete",
            {
                "turn_state": decision.turn_state,
                "transcription": decision.transcription,
                "latency_ms": decision.latency_ms,
                "audio_seconds": decision.audio_seconds,
            },
        )
        return False

    accumulated_audio_seconds = sum(
        float32_sample_count(chunk) / session.source_sample_rate
        for chunk in session.chunks
    )
    audio_data_url = await synthesize_speech_audio(session.model, EASYTURN_ACK_TEXT) if session.output_audio else None
    logger.info(
        "easyturn incomplete route: keep accumulated audio chunks=%d seconds=%.2f ack_text=%r audio=%s",
        len(session.chunks),
        accumulated_audio_seconds,
        EASYTURN_ACK_TEXT,
        bool(audio_data_url),
    )
    if session.vad is not None:
        session.vad.reset()
    session.stopped = False
    await send_event(
        session,
        "turn_incomplete",
        {
            "turn_state": decision.turn_state,
            "transcription": decision.transcription,
            "raw_output": decision.raw_output,
            "latency_ms": decision.latency_ms,
            "audio_seconds": decision.audio_seconds,
            "accumulated_chunks": len(session.chunks),
            "accumulated_audio_seconds": accumulated_audio_seconds,
            "text": EASYTURN_ACK_TEXT,
            "audio_data_url": audio_data_url,
        },
    )
    return True

async def configure_vad(session: AudioSession, value: Any) -> None:
    session.vad = None
    if not isinstance(value, dict):
        return

    engine = str(value.get("engine") or "").strip().lower()
    if engine not in {"silero", "server_silero"}:
        return
    if not SILERO_VAD_ENABLED:
        await send_event(session, "vad_error", {"message": "Silero VAD is disabled by SILERO_VAD_ENABLED=0"})
        return

    defaults = SileroVadConfig()
    config = SileroVadConfig(
        threshold=clamp_float(value.get("threshold"), default=defaults.threshold, low=0.05, high=0.95),
        min_speech_ms=clamp_int(value.get("minSpeechMs"), default=defaults.min_speech_ms, low=32, high=3000),
        min_silence_ms=clamp_int(
            value.get("minSilenceMs"),
            default=defaults.min_silence_ms,
            low=100,
            high=5000,
        ),
        max_speech_ms=clamp_int(
            value.get("maxSpeechMs"),
            default=defaults.max_speech_ms,
            low=1000,
            high=120000,
        ),
        speech_pad_ms=clamp_int(value.get("speechPadMs"), default=defaults.speech_pad_ms, low=0, high=500),
        use_onnx=parse_bool(value.get("onnx"), default=defaults.use_onnx),
    )
    try:
        session.vad = await asyncio.to_thread(SileroVadSession, config)
    except SileroVadUnavailable as exc:
        await send_event(session, "vad_error", {"message": str(exc)})
        return
    except Exception as exc:
        await send_event(session, "vad_error", {"message": f"Silero VAD load failed: {exc}"})
        return

    await send_event(
        session,
        "vad_ready",
        {
            "engine": "silero",
            "sample_rate": 16000,
            "threshold": config.threshold,
            "min_speech_ms": config.min_speech_ms,
            "min_silence_ms": config.min_silence_ms,
            "max_speech_ms": config.max_speech_ms,
        },
    )


def clamp_int(value: Any, *, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def clamp_float(value: Any, *, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "on", "yes"}


async def handle_audio_chunk(session: AudioSession, pcm_float32_bytes: bytes) -> None:
    if session.stopped:
        return
    if len(pcm_float32_bytes) < 4:
        return

    session.chunks.append(pcm_float32_bytes)
    chunk_index = len(session.chunks)
    duration_ms = int(float32_sample_count(pcm_float32_bytes) * 1000 / session.source_sample_rate)

    await send_event(
        session,
        "chunk_received",
        {
            "chunk_index": chunk_index,
            "duration_ms": duration_ms,
            "queued_prefill": PREFILL_MODE != "off",
        },
    )

    if PREFILL_MODE != "off":
        await session.prefill_queue.put((chunk_index, pcm_float32_bytes))

    if session.vad:
        await run_server_vad(session, pcm_float32_bytes)


async def handle_vad_monitor_chunk(session: AudioSession, pcm_float32_bytes: bytes) -> None:
    if session.stopped or not session.vad:
        return
    if len(pcm_float32_bytes) < 4:
        return
    await run_server_vad(session, pcm_float32_bytes)


async def run_server_vad(session: AudioSession, pcm_float32_bytes: bytes) -> None:
    try:
        vad_events = await asyncio.to_thread(
            session.vad.process_chunk,
            pcm_float32_bytes,
            session.source_sample_rate,
        )
    except Exception as exc:
        session.vad = None
        await send_event(session, "vad_error", {"message": f"Silero VAD failed: {exc}"})
        return

    for item in vad_events:
        event = item.get("event")
        if event == "speech_start":
            await send_event(session, "vad_speech_start", item)
        elif event in {"speech_end", "max_speech"}:
            await send_event(session, "vad_speech_end", item)


async def prefill_worker(session: AudioSession) -> None:
    while True:
        chunk_index, chunk = await session.prefill_queue.get()
        while not session.prefill_queue.empty():
            session.prefill_queue.task_done()
            chunk_index, chunk = await session.prefill_queue.get()
        try:
            await run_prefill_probe(session, chunk_index, chunk)
        except Exception as exc:
            await send_event(
                session,
                "prefill_error",
                {
                    "chunk_index": chunk_index,
                    "message": str(exc),
                },
            )
        finally:
            session.prefill_queue.task_done()


async def run_prefill_probe(session: AudioSession, chunk_index: int, chunk: bytes) -> None:
    start = time.perf_counter()
    if PREFILL_MODE == "cumulative_probe":
        probe_chunks = session.chunks[:chunk_index]
        prompt = (
            "这是实时语音交互中截至当前时刻的音频前缀。"
            "请只完成音频理解的预热/预填充探测，不要回答用户问题。"
            "只输出 OK。"
        )
    else:
        probe_chunks = [chunk]
        prompt = (
            "这是实时语音交互中的一个 600ms 音频 chunk。"
            "请只完成音频理解的预热/预填充探测，不要回答用户问题。"
            "只输出 OK。"
        )

    wav = wav_bytes_from_float32_chunks(probe_chunks, session.source_sample_rate, session.target_sample_rate)
    payload = build_chat_payload(
        session.model,
        wav,
        prompt,
        history=session.history,
        max_tokens=1,
        modalities=["text"],
    )

    response = await post_json(f"{QWEN_API_BASE}/chat/completions", payload)
    latency_ms = int((time.perf_counter() - start) * 1000)

    if response.status_code >= 400:
        await send_event(
            session,
            "prefill_error",
            {
                "chunk_index": chunk_index,
                "status_code": response.status_code,
                "message": response.text[:1000],
                "latency_ms": latency_ms,
            },
        )
        return

    await send_event(
        session,
        "prefill_ok",
        {
            "chunk_index": chunk_index,
            "latency_ms": latency_ms,
            "mode": PREFILL_MODE,
            "probe_chunks": len(probe_chunks),
            "note": "OpenAI-compatible prefill probe; native KV cache reuse requires server-side support.",
        },
    )


async def finalize_session(session: AudioSession) -> None:
    if not session.chunks:
        await send_event(session, "error", {"message": "no audio chunks received"})
        return

    turn_wav = audio_transcoder.float32_chunks_to_wav(session.chunks, session.source_sample_rate, session.target_sample_rate)

    tts_queue: asyncio.Queue[tuple[str, str, int] | None] = asyncio.Queue()
    tts_worker_task: asyncio.Task | None = None
    tts_index = 0

    async def tts_worker() -> None:
        while True:
            item = await tts_queue.get()
            try:
                if item is None:
                    return
                segment, text, audio_index = item
                await stream_speech_audio_events(
                    session,
                    text=text,
                    segment=segment,
                    audio_index=audio_index,
                )
            finally:
                tts_queue.task_done()

    async def enqueue_streaming_tts(segment: str, text: str) -> None:
        nonlocal tts_index, tts_worker_task
        if not session.output_audio or not session.stream_speech_audio:
            return
        tts_index += 1
        if tts_worker_task is None:
            tts_worker_task = asyncio.create_task(tts_worker())
        await tts_queue.put((segment, text, tts_index))

    try:
        state = await get_plate_agent_state(session.chat_session_id) if session.chat_session_id else PlateAgentState()
        input_wav = turn_wav
        if session.chat_session_id:
            input_wav = await build_plate_audio_input(
                session.chat_session_id,
                turn_wav,
                has_plate=state.has_car_plate,
                target_rate=session.target_sample_rate,
            )
        input_path = CAPTURE_DIR / f"{session.session_id}_input.wav"
        audio_transcoder.save_wav(input_wav, input_path)
        agent = get_plate_agent_service(model_gateway)

        async def on_ack(text: str) -> None:
            audio_url = None
            if session.output_audio and session.stream_speech_audio:
                await send_event(session, "processing_ack", {"speech_text": text, "audio_data_url": None})
                await enqueue_streaming_tts("ack", text)
                return
            elif session.output_audio:
                audio_url = await synthesize_speech_audio(session.model, text)
            await send_event(session, "processing_ack", {"speech_text": text, "audio_data_url": audio_url})

        agent_result = await agent.handle_audio_turn(
            model=session.model,
            wav_bytes=input_wav,
            state=state,
            on_ack=on_ack,
        )
    except Exception as exc:
        await send_event(
            session,
            "final_error",
            {
                "message": str(exc),
                "saved_input_wav": str(input_path) if "input_path" in locals() else "",
            },
        )
        if tts_worker_task is not None:
            await tts_queue.put(None)
            await tts_queue.join()
            await tts_worker_task
        return

    user_history_text = ""

    # Append to session history
    if session.chat_session_id:
        await append_audio_history(session.chat_session_id, turn_wav)
        await append_history(session.chat_session_id, "assistant", agent_result.history_text)
        await update_plate_agent_state(session.chat_session_id, agent_result.state)
        if not state.has_car_plate:
            if agent_result.state.has_car_plate:
                await clear_plate_audio_turns(session.chat_session_id)
            else:
                await append_plate_audio_turn(session.chat_session_id, turn_wav)

    audio_data_url = None
    if session.output_audio and not session.stream_speech_audio:
        audio_data_url = await synthesize_speech_audio(session.model, agent_result.speech_text)
    await send_event(
        session,
        "final_result",
        {
            "text": agent_result.text,
            "audio_data_url": audio_data_url,
            "history_text": agent_result.history_text,
            "speech_text": agent_result.speech_text,
            "user_history_text": user_history_text,
            "user_display_text": user_history_text,
            "output_is_json": True,
            "raw_response": {"agent_debug": agent_result.debug},
            "saved_input_wav": str(input_path),
            "latency_ms": agent_result.latency_ms,
            "ttft_ms": None,
            "agent_state": agent_result.state.to_context(),
        },
    )
    if session.output_audio and session.stream_speech_audio:
        await enqueue_streaming_tts("final", agent_result.speech_text)
        await tts_queue.join()
        await tts_queue.put(None)
        await tts_queue.join()
        if tts_worker_task is not None:
            await tts_worker_task
