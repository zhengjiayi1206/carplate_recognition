from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from realtime_audio_demo.config import QWEN_MODEL
from realtime_audio_demo.services.interfaces import ChatModel
from realtime_audio_demo.services.qwen import extract_stream_delta


logger = logging.getLogger("uvicorn.error")

PLATE_AGENT_RUNTIME_VERSION = "delta_ack_chunk_v3"

PROVINCE_ABBREVIATIONS = {
    "京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪", "苏", "浙", "皖", "闽",
    "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "渝", "川", "贵", "云", "藏",
    "陕", "甘", "青", "宁", "新",
}

CAR_PLATE_EXTRACTION_PROMPT_PATH = Path(__file__).resolve().parents[1] / "car_plate_extraction_prompt.md"
CAR_PLATE_UPDATE_PROMPT_PATH = Path(__file__).resolve().parents[1] / "car_plate_update_prompt.md"
TEMP_FEEDBACK_START_TAG = "【临时反馈】"
TEMP_FEEDBACK_END_TAG = "【/临时反馈】"
INITIAL_EXTRACT_RUNTIME_INSTRUCTION = (
    "重要输出约束：本轮是首轮车牌识别。"
    "每一段阶段性用户反馈都必须使用完整闭合标签："
    "【临时反馈】反馈内容【/临时反馈】。"
    "不能只写【临时反馈】不写【/临时反馈】；不能等到最后才闭合。"
    "每段临时反馈闭合后，才可以继续后续思考。"
    "第三阶段的临时反馈可以输出正常对话内容；"
    "第三阶段临时反馈闭合后，必须紧跟完整闭合的【车牌】车牌候选【/车牌】标签。"
    "【车牌】标签内只能输出当前最终车牌候选本身，不能带解释、标点、请确认。"
    "第四阶段的临时反馈用于向用户确认易混淆字当前识别得对不对，可以输出确认话术。"
    "后端会优先读取【车牌】标签内容做合法性校验。"
)

UPDATE_EXTRACT_RUNTIME_INSTRUCTION = (
    "重要输出约束：本轮是已有暂存车牌后的更新识别。"
    "每一段阶段性用户反馈都必须使用完整闭合标签："
    "【临时反馈】反馈内容【/临时反馈】。"
    "不能只写【临时反馈】不写【/临时反馈】；不能等到最后才闭合。"
    "每段临时反馈闭合后，才可以继续后续思考。"
    "第三阶段的临时反馈可以输出正常对话内容；"
    "第三阶段临时反馈闭合后，必须紧跟完整闭合的【车牌】修改后的车牌候选【/车牌】标签。"
    "【车牌】标签内只能输出修改后的最终车牌候选本身，不能带解释、标点、请确认。"
    "第四阶段的临时反馈用于向用户确认易混淆字当前识别得对不对，可以输出确认话术。"
    "后端会优先读取【车牌】标签内容做合法性校验。"
)


def log_node_output(node: str, output: dict[str, Any]) -> None:
    logger.info("plate_agent node=%s output=%s", node, json.dumps(output, ensure_ascii=False, default=str))


def read_prompt_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def clone_confusions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    confusions: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            confusions.append(dict(item))
    return confusions


@dataclass(slots=True)
class PlateAgentState:
    car_plate: str = ""
    vehicle_type: str = "unknown"
    confusions: list[dict[str, Any]] = field(default_factory=list)
    final_car_plate: str = ""
    assistant_reply: str = ""
    ack_sent: bool = False

    @property
    def has_car_plate(self) -> bool:
        return bool(self.car_plate)

    @property
    def is_confirmed(self) -> bool:
        return bool(self.final_car_plate)

    def to_context(self) -> dict[str, Any]:
        return {
            "plate_agent_runtime": PLATE_AGENT_RUNTIME_VERSION,
            "car_plate": self.car_plate,
            "vehicle_type": self.vehicle_type,
            "confusions": clone_confusions(self.confusions),
            "final_car_plate": self.final_car_plate,
            "ack_sent": self.ack_sent,
        }


@dataclass(slots=True)
class PlateAgentResult:
    text: str
    history_text: str
    speech_text: str
    state: PlateAgentState
    latency_ms: int
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlateValidationResult:
    length_ok: bool
    province_ok: bool
    second_char_ok: bool

    @property
    def valid(self) -> bool:
        return self.length_ok and self.province_ok and self.second_char_ok

    def to_dict(self) -> dict[str, bool]:
        return {
            "length_ok": self.length_ok,
            "province_ok": self.province_ok,
            "second_char_ok": self.second_char_ok,
        }

    def failed_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.length_ok:
            reasons.append("车牌位数必须是 7 位或 8 位")
        if not self.province_ok:
            reasons.append("车牌第一位必须是省份简称汉字")
        if not self.second_char_ok:
            reasons.append("车牌第二位必须是英文字母")
        return reasons


class PlateAgentService:
    def __init__(self, model_client: ChatModel) -> None:
        self.model_client = model_client

    async def handle_audio_turn(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        state: PlateAgentState,
        on_ack: Optional[Callable] = None,
    ) -> PlateAgentResult:
        started = time.perf_counter()
        working = clone_state(state)
        debug: dict[str, Any] = {"recognition_attempts": []}

        if working.has_car_plate and await self.detect_confirmation(model=model, wav_bytes=wav_bytes, state=working):
            working.final_car_plate = working.car_plate
            working.ack_sent = False
            assistant_reply = f"好的，已确认您的车牌号是{working.final_car_plate}。"
            working.assistant_reply = assistant_reply
            output = build_output_json(
                task_status="confirmed",
                car_plate=working.car_plate,
                assistant_reply=assistant_reply,
                final_car_plate=working.final_car_plate,
            )
            latency_ms = elapsed_ms(started)
            log_node_output(
                "turn_result",
                {
                    "stage": "confirmed",
                    "text": output,
                    "speech_text": assistant_reply,
                    "state": working.to_context(),
                    "latency_ms": latency_ms,
                },
            )
            return PlateAgentResult(
                text=output,
                history_text=output,
                speech_text=assistant_reply,
                state=working,
                latency_ms=latency_ms,
                debug=debug,
            )

        attempts: list[dict[str, Any]] = []
        had_existing_plate = working.has_car_plate
        previous_car_plate = working.car_plate

        async def emit_new_feedback(feedback: str) -> None:
            cleaned = str(feedback or "")
            if not cleaned:
                return
            await maybe_call_text_callback(on_ack, cleaned)

        raw, streamed_feedbacks = await self.stream_extract_plate_prompt_raw(
            model=model,
            wav_bytes=wav_bytes,
            previous_state=working,
            on_feedback=emit_new_feedback,
        )
        feedbacks = streamed_feedbacks or extract_temporary_feedbacks(raw)
        plate_tags = extract_plate_tags(raw)
        candidate_plate = select_plate_candidate_text(plate_tags)
        candidate_feedback = candidate_plate or select_plate_candidate_text(feedbacks) or extract_latest_temporary_feedback(raw)
        assistant_feedback = feedbacks[-1] if feedbacks else ""
        extracted_plate = extract_plate_from_feedback(raw)
        normalized_plate = normalize_plate_format(extracted_plate)
        validation = validate_plate_rules(normalized_plate)
        failed_reasons = validation.failed_reasons()
        attempts.append(
            {
                "attempt": 1,
                "raw": raw,
                "streamed_feedbacks": streamed_feedbacks,
                "plate_tags": plate_tags,
                "candidate_plate": candidate_plate,
                "candidate_feedback": candidate_feedback,
                "assistant_feedback": assistant_feedback,
                "extracted_car_plate": extracted_plate,
                "normalized_car_plate": normalized_plate,
                "previous_car_plate": previous_car_plate,
                "had_existing_plate": had_existing_plate,
                "validation": validation.to_dict(),
                "failed_reasons": failed_reasons,
            }
        )
        log_node_output("recognize_plate_from_feedback", attempts[-1])
        debug["recognition_attempts"] = attempts
        car_plate = normalized_plate

        if car_plate and validation.valid:
            working.car_plate = car_plate
            working.final_car_plate = ""
            working.vehicle_type = vehicle_type_by_length(car_plate)
            working.confusions = []
            assistant_reply = confirmation_reply_from_feedback(car_plate, assistant_feedback)
            working.assistant_reply = assistant_reply
            working.ack_sent = False
            output = build_output_json(
                task_status="need_confirmation",
                car_plate=working.car_plate,
                assistant_reply=assistant_reply,
            )
            stage = "recognized_valid_plate"
        else:
            failed_reasons = failed_reasons or ["当前车牌不满足车牌格式要求"]
            if had_existing_plate:
                assistant_reply = (
                    "这次修改后的车牌不满足车牌要求，"
                    + "；".join(failed_reasons)
                    + f"。我先保留之前识别到的车牌{clean_plate_text(previous_car_plate)}，请您重新说明要修改的部分。"
                )
                working.car_plate = previous_car_plate
                working.final_car_plate = ""
                working.vehicle_type = vehicle_type_by_length(previous_car_plate)
                output_car_plate = working.car_plate
                stage = "invalid_updated_plate_preserved_previous"
            else:
                assistant_reply = "您当前的车牌不满足车牌要求，" + "；".join(failed_reasons) + "。请您重新说一下完整车牌。"
                working.car_plate = ""
                working.final_car_plate = ""
                working.vehicle_type = "unknown"
                output_car_plate = ""
                stage = "invalid_extracted_plate"
            working.confusions = []
            working.assistant_reply = assistant_reply
            working.ack_sent = False
            output = build_output_json(
                task_status="invalid",
                car_plate=output_car_plate,
                assistant_reply=assistant_reply,
            )

        speech_text = assistant_reply
        if on_ack is None:
            speech_text = build_progressive_speech_text(feedbacks, assistant_reply)
            output = replace_output_assistant_reply(output, speech_text)

        latency_ms = elapsed_ms(started)
        log_node_output(
            "turn_result",
            {
                "stage": stage,
                "text": output,
                "speech_text": speech_text,
                "candidate_feedback": candidate_feedback,
                "state": working.to_context(),
                "latency_ms": latency_ms,
            },
        )
        return PlateAgentResult(
            text=output,
            history_text=output,
            speech_text=speech_text,
            state=working,
            latency_ms=latency_ms,
            debug=debug,
        )

    async def stream_extract_plate_prompt_raw(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        previous_state: PlateAgentState,
        on_feedback: Optional[Callable[[str], Any]] = None,
    ) -> tuple[str, list[str]]:
        if previous_state.has_car_plate:
            return await self.stream_update_plate_prompt_raw(
                model=model,
                wav_bytes=wav_bytes,
                previous_state=previous_state,
                on_feedback=on_feedback,
            )
        return await self.stream_initial_extract_plate_prompt_raw(
            model=model,
            wav_bytes=wav_bytes,
            on_feedback=on_feedback,
        )

    async def stream_initial_extract_plate_prompt_raw(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        on_feedback: Optional[Callable[[str], Any]] = None,
    ) -> tuple[str, list[str]]:
        prompt_text = read_prompt_file(CAR_PLATE_EXTRACTION_PROMPT_PATH)
        return await self.audio_stream_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=f"{prompt_text}\n\n{INITIAL_EXTRACT_RUNTIME_INSTRUCTION}",
            max_tokens=2048,
            on_feedback=on_feedback,
        )

    async def stream_update_plate_prompt_raw(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        previous_state: PlateAgentState,
        on_feedback: Optional[Callable[[str], Any]] = None,
    ) -> tuple[str, list[str]]:
        prompt_text = read_prompt_file(CAR_PLATE_UPDATE_PROMPT_PATH)
        previous_context = (
            f"\n\n当前已有暂存车牌：{clean_plate_text(previous_state.car_plate)}。"
            "本轮用户是在纠正或更新当前暂存车牌，请结合当前暂存车牌和本次音频，输出修改后的完整车牌。"
        )
        return await self.audio_stream_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=f"{prompt_text}\n\n{UPDATE_EXTRACT_RUNTIME_INSTRUCTION}{previous_context}",
            max_tokens=2048,
            on_feedback=on_feedback,
        )

    async def audio_stream_call(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        prompt: str,
        max_tokens: int,
        on_feedback: Optional[Callable[[str], Any]] = None,
    ) -> tuple[str, list[str]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def push(item: dict[str, Any]) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        stream_task = asyncio.create_task(
            self.model_client.stream_audio(
                model=model or QWEN_MODEL,
                wav_bytes=wav_bytes,
                prompt=prompt,
                history=[],
                max_tokens=max_tokens,
                response_format=None,
                push=push,
                turn_instruction="请根据这段用户语音完成当前任务。",
            )
        )
        raw_parts: list[str] = []
        streamed_feedbacks: list[str] = []
        feedback_buffer = ""
        in_feedback = False
        current_feedback = ""

        async def emit_feedback_delta(text: str) -> None:
            if not text:
                return
            if on_feedback is not None:
                await maybe_call_text_callback(on_feedback, text)

        async def append_feedback_text(text: str) -> None:
            nonlocal current_feedback
            if not text:
                return
            current_feedback += text
            await emit_feedback_delta(text)

        try:
            while True:
                if stream_task.done() and queue.empty():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue
                item_type = str(item.get("type") or "")
                if item_type == "chunk":
                    delta = extract_stream_delta(item.get("data") or {}).get("text") or ""
                    if delta:
                        raw_parts.append(delta)
                        feedback_buffer += delta
                        while feedback_buffer:
                            if in_feedback:
                                end_index = feedback_buffer.find(TEMP_FEEDBACK_END_TAG)
                                if end_index >= 0:
                                    await append_feedback_text(feedback_buffer[:end_index])
                                    feedback = current_feedback.strip()
                                    if feedback:
                                        streamed_feedbacks.append(feedback)
                                    feedback_buffer = feedback_buffer[end_index + len(TEMP_FEEDBACK_END_TAG):]
                                    in_feedback = False
                                    current_feedback = ""
                                    continue

                                safe_text, feedback_buffer = split_before_partial_tag(
                                    feedback_buffer,
                                    TEMP_FEEDBACK_END_TAG,
                                )
                                if safe_text:
                                    await append_feedback_text(safe_text)
                                break

                            start_index = feedback_buffer.find(TEMP_FEEDBACK_START_TAG)
                            if start_index >= 0:
                                feedback_buffer = feedback_buffer[start_index + len(TEMP_FEEDBACK_START_TAG):]
                                in_feedback = True
                                current_feedback = ""
                                continue

                            _, feedback_buffer = split_before_partial_tag(
                                feedback_buffer,
                                TEMP_FEEDBACK_START_TAG,
                            )
                            break
                elif item_type == "error":
                    message = str(item.get("message") or "upstream audio stream failed")
                    status_code = item.get("status_code")
                    if status_code:
                        raise RuntimeError(f"{message} (status_code={status_code})")
                    raise RuntimeError(message)
                elif item_type == "done":
                    break
            await stream_task
        except Exception:
            if not stream_task.done():
                stream_task.cancel()
            raise
        return "".join(raw_parts), streamed_feedbacks

    async def detect_confirmation(self, *, model: str, wav_bytes: bytes, state: PlateAgentState) -> bool:
        previous_ai_reply = (state.assistant_reply or "").strip()
        result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=(
                "任务：判断用户是否在确认上一轮 AI 所说的车牌信息。\n\n"
                "## 判断逻辑\n"
                "分析用户语音是否明确确认了 AI 刚才说的车牌号。"
                "用户可能说的确认话术包括：对、是的、没错、正确、就是这个、确认、嗯对、就是这样、可以了。\n"
                "用户可能说的否认话术包括：不对、修改、不是、某一位错了、听错了、不是这个、我重新说。\n\n"
                "## 输出要求\n"
                "只回答 yes 或 no，不要输出其它内容。\n"
                "yes = 用户确认了上一轮 AI 说出的车牌号\n"
                "no = 用户否认、纠正或要求修改\n\n"
                "## 上一轮 AI 对用户说的话\n"
                f"{previous_ai_reply}"
            ),
            max_tokens=2048,
        )
        confirmed = parse_yes_no(result, default=False)
        log_context = {
            "raw": result,
            "confirmed": confirmed,
            "assistant_reply": previous_ai_reply,
            "state": state.to_context(),
        }
        log_node_output("detect_confirmation", log_context)
        return confirmed

    async def audio_call(self, *, model: str, wav_bytes: bytes, prompt: str, max_tokens: int) -> str:
        completion = await self.model_client.complete_audio(
            model=model or QWEN_MODEL,
            wav_bytes=wav_bytes,
            prompt=prompt,
            history=[],
            max_tokens=max_tokens,
            turn_instruction="请根据这段用户语音完成当前任务。",
        )
        if completion.raw_response and completion.raw_response.get("status_code"):
            raise RuntimeError(str(completion.raw_response.get("message") or "upstream audio request failed"))
        return completion.text or ""


def clone_state(state: PlateAgentState) -> PlateAgentState:
    return PlateAgentState(
        car_plate=state.car_plate,
        vehicle_type=state.vehicle_type,
        confusions=clone_confusions(state.confusions),
        final_car_plate=state.final_car_plate,
        assistant_reply=state.assistant_reply,
        ack_sent=state.ack_sent,
    )


def build_output_json(
    *,
    task_status: str,
    car_plate: str,
    assistant_reply: str,
    final_car_plate: str = "",
) -> str:
    data: dict[str, Any] = {
        "task_status": task_status,
        "car_plate": clean_plate_text(car_plate),
        "assistant_reply": assistant_reply,
    }
    if final_car_plate:
        data["final_plate_number"] = clean_plate_text(final_car_plate)
    return json.dumps(data, ensure_ascii=False, indent=2)


def extract_latest_temporary_feedback(text: Any) -> str:
    raw = str(text or "")
    matches = extract_temporary_feedbacks(raw)
    return matches[-1] if matches else raw.strip()


def extract_temporary_feedbacks(text: Any) -> list[str]:
    raw = str(text or "")
    return [str(match).strip() for match in re.findall(r"【临时反馈】(.*?)【/临时反馈】", raw, flags=re.S)]


def extract_plate_tags(text: Any) -> list[str]:
    raw = str(text or "")
    return [str(match).strip() for match in re.findall(r"【车牌】(.*?)【/车牌】", raw, flags=re.S)]


def select_plate_candidate_text(values: list[str]) -> str:
    for value in reversed(values):
        normalized = normalize_plate_format(value)
        if normalized and validate_plate_rules(normalized).valid:
            return value
    return ""


def extract_plate_from_feedback(text: Any) -> str:
    raw = str(text or "")
    candidate_plate = select_plate_candidate_text(extract_plate_tags(raw))
    if candidate_plate:
        return candidate_plate

    feedbacks = extract_temporary_feedbacks(raw)
    candidate_feedback = select_plate_candidate_text(feedbacks)
    if candidate_feedback:
        return candidate_feedback

    patterns = [
        r"您说的车牌是[:：]?\s*(.+?)(?:[，。；\n]|请确认|请您确认|$)",
        r"我识别到的车牌是[:：]?\s*(.+?)(?:[，。；\n]|请确认|请您确认|$)",
        r"当前识别的车牌是[:：]?\s*(.+?)(?:[，。；\n]|请确认|请您确认|$)",
        r"识别到的车牌是[:：]?\s*(.+?)(?:[，。；\n]|请确认|请您确认|$)",
        r"当前识别为[:：]?\s*(.+?)(?:[，。；\n]|请确认|请您确认|$)",
        r"当前最终车牌[:：]\s*(.+?)(?:[，。；\n]|请确认|请您确认|$)",
        r"最终车牌[:：]\s*(.+?)(?:[，。；\n]|请确认|请您确认|$)",
    ]
    for source in [*reversed(feedbacks), raw]:
        for pattern in patterns:
            match = re.search(pattern, source)
            if match:
                return str(match.group(1)).strip()
    return extract_latest_temporary_feedback(raw)


def confirmation_reply_from_feedback(car_plate: str, feedback: str) -> str:
    cleaned = str(feedback or "").strip()
    if cleaned and not select_plate_candidate_text([cleaned]):
        return cleaned
    return confirmation_reply_for_plate(car_plate)


def confirmation_reply_for_plate(car_plate: str) -> str:
    return f"您说的车牌是：{clean_plate_text(car_plate)}，请您确认是否正确。"


def build_progressive_speech_text(feedbacks: list[str], final_text: str) -> str:
    parts: list[str] = []
    for value in [*feedbacks, final_text]:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        if parts and cleaned == parts[-1]:
            continue
        parts.append(cleaned)
    return "\n".join(parts)


def replace_output_assistant_reply(output: str, assistant_reply: str) -> str:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return output
    if not isinstance(data, dict):
        return output
    data["assistant_reply"] = assistant_reply
    return json.dumps(data, ensure_ascii=False, indent=2)


def split_before_partial_tag(text: str, tag: str) -> tuple[str, str]:
    max_suffix_length = min(len(text), len(tag) - 1)
    for suffix_length in range(max_suffix_length, 0, -1):
        suffix = text[-suffix_length:]
        if tag.startswith(suffix):
            return text[:-suffix_length], suffix
    return text, ""


def validate_plate_rules(car_plate: str) -> PlateValidationResult:
    plate = normalize_plate_format(car_plate)
    length_ok = len(plate) in {7, 8}
    province_ok = bool(plate) and plate[0] in PROVINCE_ABBREVIATIONS
    second_char_ok = len(plate) >= 2 and plate[1].isascii() and plate[1].isalpha()
    return PlateValidationResult(
        length_ok=length_ok,
        province_ok=province_ok,
        second_char_ok=second_char_ok,
    )


def parse_yes_no(text: str, *, default: bool) -> bool:
    value = str(text or "").strip().lower()
    if re.search(r"\byes\b", value) or "true" in value:
        return True
    if re.search(r"\bno\b", value) or "false" in value:
        return False
    return default


def clean_plate_text(value: Any) -> str:
    return re.sub(r"[\s\-_,，。；;:：]+", "", str(value or "")).strip()


def normalize_plate_format(value: Any) -> str:
    return clean_plate_text(value).upper()


def plate_length(car_plate: str) -> int:
    return len(clean_plate_text(car_plate))


def vehicle_type_by_length(car_plate: str) -> str:
    length = plate_length(car_plate)
    if length == 7:
        return "fuel"
    if length == 8:
        return "new_energy"
    return "unknown"


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


async def maybe_call_text_callback(callback: Optional[Callable[[str], Any]], text: str) -> None:
    if callback is None:
        return
    result = callback(text)
    if inspect.isawaitable(result):
        await result


plate_agent_service: PlateAgentService | None = None


def get_plate_agent_service(model_client: ChatModel) -> PlateAgentService:
    global plate_agent_service
    if plate_agent_service is None or plate_agent_service.model_client is not model_client:
        plate_agent_service = PlateAgentService(model_client)
    return plate_agent_service
