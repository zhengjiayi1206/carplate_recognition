import asyncio
import base64
import logging
from typing import Any

from realtime_audio_demo.config import FINAL_MAX_TOKENS, QWEN_MODEL, normalize_model_name
from realtime_audio_demo.services.interfaces import ChatModel, SpeechSynthesizer
from realtime_audio_demo.services.final_plate import resolve_confirmed_plate_output
from realtime_audio_demo.services.model_gateway import model_gateway
from realtime_audio_demo.services.output_filter import normalize_assistant_output, validate_current_normalized_plate
from realtime_audio_demo.services.plate_agent import PlateAgentState, get_plate_agent_service
from realtime_audio_demo.services.prompt_provider import system_prompt_provider
from realtime_audio_demo.services.qwen import normalize_history
from realtime_audio_demo.services.speech import speech_synthesizer
from realtime_audio_demo.session_store import (
    append_audio_history,
    append_history,
    get_plate_agent_state,
    get_session_history,
    reset_session_state,
    update_plate_agent_state,
)


SESSION_OPENING_TEXT = "您好，请告诉我您的车牌号。"
PREFIX_WARMUP_TEXT = "请只进行会话前缀预热，不要回答用户问题。只输出 OK。"
logger = logging.getLogger("uvicorn.error")


class ChatboxApplicationService:
    def __init__(self, model_client: ChatModel, speech: SpeechSynthesizer) -> None:
        self.model_client = model_client
        self.speech = speech

    async def start_session(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        model = normalize_model_name(payload.get("model") or QWEN_MODEL)
        session_id = str(payload.get("session_id") or "").strip()
        audio_data_url = None
        if bool(payload.get("outputAudio")):
            audio_data_url = await self.speech.synthesize(model=model, text=SESSION_OPENING_TEXT)

        if session_id:
            await reset_session_state(session_id)
            await append_history(session_id, "assistant", SESSION_OPENING_TEXT)

        asyncio.create_task(self.warmup_session_prefix(model, session_id))

        return {
            "session_id": session_id or None,
            "text": SESSION_OPENING_TEXT,
            "audio_data_url": audio_data_url,
            "history_text": SESSION_OPENING_TEXT,
            "speech_text": SESSION_OPENING_TEXT,
            "output_is_json": False,
        }, 200

    async def warmup_session_prefix(self, model: str, session_id: str) -> None:
        try:
            history = normalize_history(await get_session_history(session_id)) if session_id else [
                {"role": "assistant", "content": SESSION_OPENING_TEXT}
            ]
            await self.model_client.complete_text(
                model=model,
                text=PREFIX_WARMUP_TEXT,
                prompt=system_prompt_provider.get_prompt(),
                history=history,
                max_tokens=1,
                output_audio=False,
            )
        except Exception as exc:
            logger.warning("session prefix warmup failed: %s", exc)

    async def handle_text(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        user_text = str(payload.get("text") or "").strip()
        if not user_text:
            return {"message": "text is required"}, 400

        model = normalize_model_name(payload.get("model") or QWEN_MODEL)
        session_id = str(payload.get("session_id") or "").strip()
        history = normalize_history(await get_session_history(session_id)) if session_id else normalize_history(payload.get("history"))
        result, status_code = await self.model_client.complete_text(
            model=model,
            text=user_text,
            prompt=system_prompt_provider.get_prompt(),
            history=history,
            max_tokens=FINAL_MAX_TOKENS,
            output_audio=False,
        )
        if status_code >= 400:
            return result, status_code

        checked_text = validate_current_normalized_plate(result.get("text"))
        final_text = await resolve_confirmed_plate_output(
            model_client=self.model_client,
            model=model,
            assistant_text=checked_text,
        )
        result["text"] = final_text
        normalized = normalize_assistant_output(final_text)
        result["text"] = normalized.display_text
        result["history_text"] = normalized.history_text
        result["speech_text"] = normalized.speech_text
        result["output_is_json"] = normalized.is_json
        result["session_id"] = session_id or None

        if session_id:
            await append_history(session_id, "user", user_text)
            await append_history(session_id, "assistant", normalized.history_text)

        return result, 200

    async def synthesize_speech(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = str(payload.get("text") or "").strip()
        if not text:
            return {"audio_data_url": None}
        model = normalize_model_name(payload.get("model") or QWEN_MODEL)
        return {"audio_data_url": await self.speech.synthesize(model=model, text=text)}

    async def handle_audio(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        audio_base64 = str(payload.get("audio_base64") or "").strip()
        if not audio_base64:
            return {"message": "audio_base64 is required"}, 400
        if "," in audio_base64 and audio_base64.startswith("data:"):
            audio_base64 = audio_base64.split(",", 1)[1]
        try:
            wav_bytes = base64.b64decode(audio_base64, validate=True)
        except Exception as exc:
            return {"message": f"bad audio_base64: {exc}"}, 400

        model = normalize_model_name(payload.get("model") or QWEN_MODEL)
        session_id = str(payload.get("session_id") or "").strip()
        state = await get_plate_agent_state(session_id) if session_id else PlateAgentState()
        try:
            agent = get_plate_agent_service(self.model_client)
            agent_result = await agent.handle_audio_turn(
                model=model,
                wav_bytes=wav_bytes,
                state=state,
            )
        except Exception as exc:
            return {"message": f"upstream audio request failed: {exc}"}, 502
        if session_id:
            await append_audio_history(session_id, wav_bytes)
            await append_history(session_id, "assistant", agent_result.history_text)
            await update_plate_agent_state(session_id, agent_result.state)

        audio_data_url = None
        if bool(payload.get("outputAudio")):
            audio_data_url = await self.speech.synthesize(model=model, text=agent_result.speech_text)

        return {
            "text": agent_result.text,
            "audio_data_url": audio_data_url,
            "history_text": agent_result.history_text,
            "speech_text": agent_result.speech_text,
            "user_history_text": "",
            "user_display_text": "",
            "output_is_json": True,
            "latency_ms": agent_result.latency_ms,
            "ttft_ms": None,
            "agent_state": agent_result.state.to_context(),
            "agent_debug": agent_result.debug,
        }, 200

chatbox_service = ChatboxApplicationService(model_gateway, speech_synthesizer)
