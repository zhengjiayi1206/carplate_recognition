from __future__ import annotations

import json
import time
from typing import Any

from realtime_audio_demo.config import PLATE_REPLY_INCLUDE_CONFIRMATION, QWEN_MODEL
from realtime_audio_demo.services.interfaces import ChatModel
from realtime_audio_demo.services.plate_agent_ack import ack_schedule_for_state
from realtime_audio_demo.services.plate_agent_confirmation import apply_confirmation_actions
from realtime_audio_demo.services.plate_agent_logging import (
    CURRENT_SESSION_ID,
    CURRENT_TURN_BEFORE_STATE,
    log_agent_line,
    log_node_output,
    log_session_event,
    logger,
)
from realtime_audio_demo.services.plate_agent_messages import (
    EDIT_UNCLEAR_REPLY,
    INVALID_PLATE_REPLY,
    NO_PLATE_REPLY,
    build_confirmed_reply,
    build_edit_invalid_reply,
    build_fixed_reply,
)
from realtime_audio_demo.services.plate_agent_parsing import elapsed_ms
from realtime_audio_demo.services.plate_agent_prompts import (
    build_plate_agent_status_bar,
    build_plate_agent_system_prompt,
)
from realtime_audio_demo.services.plate_agent_response import build_output_json, reply_with_pending_confirmation
from realtime_audio_demo.services.plate_agent_rules import is_valid_plate_number
from realtime_audio_demo.services.plate_agent_state import clone_state
from realtime_audio_demo.services.plate_agent_tooling import (
    PLATE_WRITE_TOOL_NAMES,
    PlateAgentPlan,
    PlateToolExecutor,
    build_assistant_tool_call_message,
    build_compatible_observation_message,
    build_tool_observation_messages,
    parse_agent_plan,
)
from realtime_audio_demo.services.plate_agent_types import (
    PlateAgentResult,
    PlateAgentState,
    PlateConfirmationAction,
)


MAX_AGENT_ITERATIONS = 8


class PlateAgentService:
    """车牌语音 Agent 主服务。

    主流程只负责搭建 Agent 循环：
    1. 注入后端维护的状态栏。
    2. 让模型输出 tool_calls 或 finish。
    3. 后端执行工具并把 observation 回填到上下文。
    4. 根据最终状态生成接口需要的 JSON 和播报话术。
    """

    def __init__(self, model_client: ChatModel) -> None:
        self.model_client = model_client

    async def handle_audio_turn(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        state: PlateAgentState,
        session_id: str = "",
        on_ack: Any = None,
        turn_summaries: list[str] | None = None,
        include_confirmation_reply: bool = PLATE_REPLY_INCLUDE_CONFIRMATION,
    ) -> PlateAgentResult:
        started = time.perf_counter()
        working = clone_state(state)
        if turn_summaries is not None:
            working.turn_summaries = list(turn_summaries)[-6:]

        before_state = working.to_context()
        CURRENT_SESSION_ID.set(str(session_id or "").strip())
        CURRENT_TURN_BEFORE_STATE.set(before_state)
        log_node_output(
            "handle_audio_turn.start",
            {
                "action": "start_agent_audio_turn",
                "model": model or QWEN_MODEL,
                "wav_bytes": len(wav_bytes),
                "before_state": before_state,
                "state": working.to_context(),
            },
        )
        await self.emit_compat_ack_if_needed(on_ack=on_ack, state=working)

        log_agent_line(
            "Agent 回合开始",
            阶段="多轮确认或纠错" if working.has_car_plate else "首轮识别",
            当前状态=working.to_context(),
            说明="模型先根据本轮音频和初始状态栏规划，后续根据工具 observation 自主继续。",
        )

        executor = PlateToolExecutor(working)
        observations: list[dict[str, Any]] = []
        agent_history: list[dict[str, Any]] = []
        plans: list[dict[str, Any]] = []
        last_plan = PlateAgentPlan(raw="")

        for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
            raw_plan, request_status = await self.plan_next_action(
                model=model,
                wav_bytes=wav_bytes,
                state=working,
                iteration=iteration,
                observations=observations,
                agent_history=agent_history,
            )
            last_plan = parse_agent_plan(raw_plan)
            plans.append(last_plan.to_dict())
            if iteration == 1:
                agent_history.append(build_audio_user_history_message(wav_bytes))
            if last_plan.tool_calls:
                agent_history.append(build_assistant_tool_call_message(last_plan))
            else:
                agent_history.append({"role": "assistant", "content": raw_plan})
            log_session_event(
                "llm_response",
                iteration=iteration,
                raw_output=raw_plan,
                parsed_plan=last_plan.to_dict(),
                agent_history=compact_agent_history(agent_history),
                state=working.to_context(),
            )
            log_agent_line(
                "Agent 规划：模型推理是什么",
                Agent循环序号=iteration,
                模型输出=raw_plan,
            )
            log_agent_line(
                "Agent 规划：tool_calls 是什么",
                Agent循环序号=iteration,
                thought=last_plan.thought,
                tool_calls=[item.to_dict() for item in last_plan.tool_calls],
                finish=last_plan.finish,
            )
            log_node_output(
                "agent.plan",
                {
                    "iteration": iteration,
                    "raw": raw_plan,
                    "plan": last_plan.to_dict(),
                    "state": working.to_context(),
                    "observations": observations,
                    "agent_history": compact_agent_history(agent_history),
                },
            )

            if last_plan.tool_calls:
                current_observations = executor.execute_all(last_plan.tool_calls)
                observations.extend(current_observations)
                agent_history.extend(build_tool_observation_messages(current_observations))
                continue

            if last_plan.finish:
                break

            log_agent_line(
                "Agent 规划为空",
                Agent循环序号=iteration,
                说明="模型没有输出可执行工具，也没有输出 finish，结束循环并走兜底回复。",
            )
            break

        return self.build_final_result(
            started=started,
            before_state=before_state,
            working=working,
            last_plan=last_plan,
            plans=plans,
            observations=observations,
            agent_history=agent_history,
            include_confirmation_reply=include_confirmation_reply,
        )

    async def emit_compat_ack_if_needed(self, *, on_ack: Any, state: PlateAgentState) -> None:
        """兼容旧调用方：如果还传 on_ack，只发送第一条衔接语。"""

        if on_ack is None:
            return
        try:
            _, ack_text = ack_schedule_for_state(state)[0]
            await on_ack(ack_text)
            log_node_output(
                "handle_audio_turn.compat_on_ack",
                {
                    "action": "emit_compat_ack",
                    "ack_text": ack_text,
                    "state": state.to_context(),
                },
            )
        except Exception as exc:
            logger.warning("plate_agent compat on_ack failed: %s", exc)

    async def plan_next_action(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        state: PlateAgentState,
        iteration: int,
        observations: list[dict[str, Any]],
        agent_history: list[dict[str, Any]],
    ) -> tuple[str, str]:
        """调用模型，让模型根据音频、状态栏或 observation 输出 tool_calls / finish。"""

        system_prompt = build_plate_agent_system_prompt()
        status_bar = ""
        if iteration == 1:
            status_bar = build_plate_agent_status_bar(
                state=state,
            )
        turn_instruction = status_bar if iteration == 1 else ""
        request_history = list(agent_history)
        log_session_event(
            "llm_request",
            iteration=iteration,
            model=model or QWEN_MODEL,
            input_type="audio" if iteration == 1 else "text",
            audio_bytes=len(wav_bytes) if iteration == 1 else 0,
            status_bar=status_bar,
            turn_instruction=turn_instruction,
            agent_history=compact_agent_history(agent_history),
            state=state.to_context(),
            observations=observations,
        )
        if iteration == 1:
            completion = await self.model_client.complete_audio(
                model=model or QWEN_MODEL,
                wav_bytes=wav_bytes,
                prompt=system_prompt,
                history=request_history,
                max_tokens=1024,
                turn_instruction=turn_instruction,
            )
            if completion.raw_response and completion.raw_response.get("status_code"):
                completion = await self.retry_audio_with_compatible_history(
                    model=model,
                    wav_bytes=wav_bytes,
                    system_prompt=system_prompt,
                    agent_history=agent_history,
                    turn_instruction=turn_instruction,
                )
            return completion.text or "", status_bar

        response, status_code = await self.model_client.complete_text(
            model=model or QWEN_MODEL,
            text=turn_instruction,
            prompt=system_prompt,
            history=request_history,
            max_tokens=1024,
            output_audio=False,
        )
        if status_code >= 400:
            response, status_code = await self.retry_text_with_compatible_history(
                model=model,
                system_prompt=system_prompt,
                agent_history=agent_history,
                turn_instruction=turn_instruction,
            )
        if status_code >= 400:
            raise RuntimeError(str(response.get("message") or "upstream text request failed"))
        return str(response.get("text") or ""), status_bar

    async def retry_audio_with_compatible_history(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        system_prompt: str,
        agent_history: list[dict[str, Any]],
        turn_instruction: str,
    ) -> Any:
        """部分 Qwen/OpenAI 兼容服务不接受 role=tool，失败时降级成文本 observation。"""

        logger.warning("plate_agent retry audio with compatible observation history")
        completion = await self.model_client.complete_audio(
            model=model or QWEN_MODEL,
            wav_bytes=wav_bytes,
            prompt=system_prompt,
            history=compatible_agent_history(agent_history),
            max_tokens=1024,
            turn_instruction=turn_instruction,
        )
        if completion.raw_response and completion.raw_response.get("status_code"):
            raise RuntimeError(str(completion.raw_response.get("message") or "upstream audio request failed"))
        return completion

    async def retry_text_with_compatible_history(
        self,
        *,
        model: str,
        system_prompt: str,
        agent_history: list[dict[str, Any]],
        turn_instruction: str,
    ) -> tuple[dict[str, Any], int]:
        """文本续轮的兼容降级：把 tool role observation 转成 user 文本。"""

        logger.warning("plate_agent retry text with compatible observation history")
        return await self.model_client.complete_text(
            model=model or QWEN_MODEL,
            text=turn_instruction,
            prompt=system_prompt,
            history=compatible_agent_history(agent_history),
            max_tokens=1024,
            output_audio=False,
        )

    def build_final_result(
        self,
        *,
        started: float,
        before_state: dict[str, Any],
        working: PlateAgentState,
        last_plan: PlateAgentPlan,
        plans: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        agent_history: list[dict[str, Any]],
        include_confirmation_reply: bool = PLATE_REPLY_INCLUDE_CONFIRMATION,
    ) -> PlateAgentResult:
        """把 Agent 最终状态转成前端接口仍然兼容的输出格式。"""

        finish_status = normalize_finish_status(last_plan.finish.get("task_status"))
        reply_scene = str(last_plan.finish.get("reply_scene") or "").strip()
        failed_result = last_failed_observation(observations)
        state_changed = agent_changed_state(before_state, working)

        if finish_status == "confirmed" and can_confirm_from_finish(working, observations):
            confirm_current_plate_from_finish(working)

        if finish_status == "confirmed" and working.final_car_plate:
            assistant_reply = build_confirmed_reply(working.final_car_plate)
            task_status = "confirmed"
        elif not working.has_car_plate:
            task_status, assistant_reply = self.reply_without_plate(
                finish_status=finish_status,
                failed_result=failed_result,
            )
        elif finish_status == "invalid" and failed_result is not None:
            task_status = "need_confirmation"
            assistant_reply = build_edit_invalid_reply(working, include_confirmation=include_confirmation_reply)
        elif (not last_plan.finish or finish_status == "unclear" or failed_result is not None) and not state_changed:
            task_status = "need_confirmation"
            assistant_reply = self.reply_for_failed_or_unclear_edit(
                working,
                failed_result,
                include_confirmation_reply=include_confirmation_reply,
            )
        else:
            task_status = "need_confirmation"
            assistant_reply = self.reply_for_current_state(
                working,
                before_state=before_state,
                reply_scene=reply_scene,
                include_confirmation_reply=include_confirmation_reply,
            )

        working.assistant_reply = assistant_reply
        working.ack_sent = False
        output = build_output_json(
            task_status=task_status,
            car_plate=working.car_plate,
            assistant_reply=assistant_reply,
            final_car_plate=working.final_car_plate if task_status == "confirmed" else "",
        )
        latency_ms = elapsed_ms(started)
        debug = {
            "agent_plans": plans,
            "observations": observations,
            "agent_history": compact_agent_history(agent_history),
            "finish": last_plan.finish,
            "car_plate": working.car_plate,
            "vehicle_type": working.vehicle_type,
        }
        log_node_output(
            "turn_result",
            {
                "stage": task_status,
                "text": output,
                "speech_text": assistant_reply,
                "state": working.to_context(),
                "latency_ms": latency_ms,
                "agent_plans": plans,
                "observations": observations,
            },
        )
        log_session_event(
            "final_response",
            task_status=task_status,
            response_text=output,
            speech_text=assistant_reply,
            latency_ms=latency_ms,
            state=working.to_context(),
            agent_plans=plans,
            observations=observations,
            agent_history=compact_agent_history(agent_history),
        )
        return PlateAgentResult(
            text=output,
            history_text=output,
            speech_text=assistant_reply,
            state=working,
            latency_ms=latency_ms,
            debug=debug,
        )

    def reply_without_plate(
        self,
        *,
        finish_status: str,
        failed_result: dict[str, Any] | None,
    ) -> tuple[str, str]:
        if finish_status == "invalid" or failed_tool_name(failed_result) == "set_plate":
            return "invalid", INVALID_PLATE_REPLY
        return "need_more_info", NO_PLATE_REPLY

    def reply_for_failed_or_unclear_edit(
        self,
        working: PlateAgentState,
        failed_result: dict[str, Any] | None,
        *,
        include_confirmation_reply: bool,
    ) -> str:
        message = str((failed_result or {}).get("message") or "").strip()
        if message and "格式不合法" in message:
            return build_edit_invalid_reply(working, include_confirmation=include_confirmation_reply)
        return reply_with_pending_confirmation(
            message or EDIT_UNCLEAR_REPLY,
            working,
            include_confirmation=include_confirmation_reply,
        )

    def reply_for_current_state(
        self,
        working: PlateAgentState,
        *,
        before_state: dict[str, Any],
        reply_scene: str,
        include_confirmation_reply: bool,
    ) -> str:
        if reply_scene in {"initial_success", "update_success", "partial_confirmation"}:
            return build_fixed_reply(
                working,
                changed=reply_scene == "update_success",
                scene=reply_scene,
                include_confirmation=include_confirmation_reply,
            )
        before_plate = str(before_state.get("car_plate") or "").strip()
        if not before_plate and working.car_plate:
            return build_fixed_reply(
                working,
                changed=True,
                scene="initial_success",
                include_confirmation=include_confirmation_reply,
            )
        if before_plate and before_plate != working.car_plate:
            return build_fixed_reply(
                working,
                changed=True,
                scene="update_success",
                include_confirmation=include_confirmation_reply,
            )
        return build_fixed_reply(
            working,
            changed=False,
            scene="partial_confirmation",
            include_confirmation=include_confirmation_reply,
        )


def normalize_finish_status(value: Any) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "need_more_info": "need_more_info",
        "more_info": "need_more_info",
        "need_confirmation": "need_confirmation",
        "confirmation": "need_confirmation",
        "confirmed": "confirmed",
        "success": "confirmed",
        "invalid": "invalid",
        "unclear": "unclear",
        "unknown": "unclear",
    }
    return aliases.get(raw, "")


def last_failed_observation(observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in reversed(observations):
        if item.get("success") is False:
            return item
    return None


def agent_changed_state(
    before_state: dict[str, Any],
    working: PlateAgentState,
) -> bool:
    current_state = working.to_context()
    for key in ("car_plate", "confirmed", "final_car_plate", "need_confirm_chars", "confirmed_chars"):
        if before_state.get(key) != current_state.get(key):
            return True
    return False


def can_confirm_from_finish(
    working: PlateAgentState,
    observations: list[dict[str, Any]],
) -> bool:
    if working.final_car_plate:
        return True
    if not working.car_plate or not is_valid_plate_number(working.car_plate):
        return False
    return not turn_wrote_plate(observations)


def confirm_current_plate_from_finish(working: PlateAgentState) -> None:
    apply_confirmation_actions(
        working,
        [PlateConfirmationAction(action="confirm_all")],
        source="finish.confirmed",
    )
    working.final_car_plate = working.car_plate
    working.confirmed = True


def turn_wrote_plate(observations: list[dict[str, Any]]) -> bool:
    for item in observations:
        if item.get("success") is True and str(item.get("name") or "").strip() in PLATE_WRITE_TOOL_NAMES:
            return True
    return False


def failed_tool_name(result: dict[str, Any] | None) -> str:
    return str((result or {}).get("name") or "").strip()


def build_audio_user_history_message(wav_bytes: bytes) -> dict[str, Any]:
    """续轮上下文只保留用户音频占位，避免重复注入首轮状态栏。"""

    return {
        "role": "user",
        "content": f"<user_audio>本轮用户音频已输入，字节数：{len(wav_bytes)}。</user_audio>",
    }


def compatible_agent_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把标准 Agent 轨迹降级为只含 user/assistant 的文本轨迹。"""

    compatible: list[dict[str, Any]] = []
    pending_observations: list[dict[str, Any]] = []
    for item in history:
        role = item.get("role")
        if role == "tool":
            pending_observations.append(tool_message_to_observation(item))
            continue
        flush_pending_observations(compatible, pending_observations)
        if role == "system":
            compatible.append({"role": "user", "content": str(item.get("content") or "")})
        elif role == "assistant" and item.get("tool_calls"):
            compatible.append({"role": "assistant", "content": assistant_tool_call_text(item)})
        elif role in {"user", "assistant"}:
            content = item.get("content")
            if content:
                compatible.append({"role": role, "content": content})
    flush_pending_observations(compatible, pending_observations)
    return compatible


def flush_pending_observations(history: list[dict[str, Any]], observations: list[dict[str, Any]]) -> None:
    if not observations:
        return
    history.append(build_compatible_observation_message(list(observations)))
    observations.clear()


def tool_message_to_observation(item: dict[str, Any]) -> dict[str, Any]:
    raw_content = item.get("content")
    if isinstance(raw_content, str):
        try:
            content = json.loads(raw_content)
        except json.JSONDecodeError:
            content = {"message": raw_content}
    else:
        content = raw_content if isinstance(raw_content, dict) else {}
        return {
            "tool_call_id": item.get("tool_call_id"),
            "name": item.get("name") or content.get("name"),
            "success": content.get("success"),
            "message": content.get("message"),
            "arguments": content.get("arguments"),
            "data": content.get("data"),
            "after_state": content.get("after_state"),
        }


def assistant_tool_call_text(item: dict[str, Any]) -> str:
    calls: list[dict[str, Any]] = []
    for call in item.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, dict) and isinstance(call.get("function"), dict) else {}
        arguments = function.get("arguments")
        try:
            parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            parsed_arguments = {}
        calls.append(
            {
                "name": function.get("name"),
                "arguments": parsed_arguments or {},
            }
        )
    return json.dumps({"thought": item.get("content") or "", "tool_calls": calls}, ensure_ascii=False)


def compact_agent_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    compacted: list[dict[str, str]] = []
    for item in history[-10:]:
        role = str(item.get("role") or "").strip()
        if item.get("tool_calls"):
            content = assistant_tool_call_text(item)
        else:
            content = str(item.get("content") or "").strip()
        if not role or not content:
            continue
        compacted.append({"role": role, "content": content[:3000]})
    return compacted


plate_agent_service: PlateAgentService | None = None


def get_plate_agent_service(model_client: ChatModel) -> PlateAgentService:
    """返回单例 PlateAgentService，避免每个请求重复创建服务对象。"""

    global plate_agent_service
    if plate_agent_service is None or plate_agent_service.model_client is not model_client:
        plate_agent_service = PlateAgentService(model_client)
    return plate_agent_service
