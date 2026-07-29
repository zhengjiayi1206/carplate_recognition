from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from realtime_audio_demo.services.output_filter import extract_json_candidate
from realtime_audio_demo.services.plate_agent_confirmation import (
    apply_confirmation_actions,
)
from realtime_audio_demo.services.plate_agent_edit import (
    apply_plate_edit_command,
    normalize_edit_value,
    normalize_occurrence,
    normalize_relation,
    parse_positive_int,
)
from realtime_audio_demo.services.plate_agent_logging import log_agent_line, log_node_output, log_session_event
from realtime_audio_demo.services.plate_agent_parsing import sanitize_extracted_plate_text, unique_positions
from realtime_audio_demo.services.plate_agent_rules import (
    clean_plate_text,
    detect_initial_confusions_by_rule,
    is_valid_plate_number,
    normalize_plate_text,
    plate_length,
    vehicle_type_by_length,
)
from realtime_audio_demo.services.plate_agent_state import refresh_plate_state
from realtime_audio_demo.services.plate_agent_types import (
    PlateAgentState,
    PlateConfirmationAction,
    PlateEditCommand,
    PlateEditResult,
)


EDIT_TOOL_NAMES = {"replace_position", "replace_char", "insert_position", "delete_position"}
PLATE_WRITE_TOOL_NAMES = {"set_plate", *EDIT_TOOL_NAMES}
CONFIRMATION_TOOL_NAMES = {
    "validate_plate_rules",
    "detect_confusions_by_rules",
    "remove_need_confirmation",
    "add_confirmed",
    "remove_confirmed",
    "confirm_all",
}
AGENT_TOOL_NAMES = {
    "get_current_state",
    "set_plate",
    *EDIT_TOOL_NAMES,
    *CONFIRMATION_TOOL_NAMES,
}


@dataclass(slots=True)
class PlateToolCall:
    """模型输出的一次工具调用。"""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.tool_call_id,
            "name": self.name,
            "arguments": self.arguments,
            "raw": self.raw,
        }


@dataclass(slots=True)
class PlateAgentPlan:
    """模型本轮规划结果。"""

    raw: str
    thought: str = ""
    tool_calls: list[PlateToolCall] = field(default_factory=list)
    finish: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "thought": self.thought,
            "tool_calls": [item.to_dict() for item in self.tool_calls],
            "finish": self.finish,
        }


def parse_agent_plan(text: Any) -> PlateAgentPlan:
    """从模型输出里取最后一个 JSON，并兼容 OpenAI tool_calls 形状。"""

    raw = str(text or "").strip()
    data = parse_json_object(raw)
    thought = str(
        data.get("thought")
        or data.get("analysis")
        or data.get("reason")
        or data.get("reasoning")
        or ""
    ).strip()
    calls = [parse_tool_call(item, index) for index, item in enumerate(extract_tool_call_items(data), start=1)]
    finish = data.get("finish") if isinstance(data.get("finish"), dict) else {}
    return PlateAgentPlan(
        raw=raw,
        thought=thought,
        tool_calls=[item for item in calls if item is not None],
        finish=finish,
    )


def parse_json_object(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(extract_json_candidate(raw, prefer_object=True))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def extract_tool_call_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("tool_calls", "tools", "calls", "actions"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if any(key in data for key in ("name", "tool", "function")):
        return [data]
    return []


def parse_tool_call(data: dict[str, Any], index: int) -> PlateToolCall | None:
    function_data = data.get("function") if isinstance(data.get("function"), dict) else None
    name = ""
    arguments: dict[str, Any] = {}
    if function_data is not None:
        name = str(function_data.get("name") or data.get("name") or "").strip()
        arguments = parse_arguments(function_data.get("arguments"))
    else:
        name = str(data.get("name") or data.get("tool") or data.get("function") or "").strip()
        arguments = parse_arguments(data.get("arguments"))
        if not arguments and "arguments" not in data:
            arguments = {key: value for key, value in data.items() if key not in {"name", "tool", "function", "id"}}

    normalized_name = normalize_tool_name(name)
    if normalized_name not in AGENT_TOOL_NAMES:
        return None
    return PlateToolCall(
        name=normalized_name,
        arguments=arguments,
        tool_call_id=str(data.get("id") or f"call_{index}"),
        raw=data,
    )


def parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(extract_json_candidate(value, prefer_object=True))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def normalize_tool_name(value: Any) -> str:
    raw = str(value or "").strip()
    aliases = {
        "replace_at_position": "replace_position",
        "replace_by_position": "replace_position",
        "replace_symbol": "replace_char",
        "replace_value": "replace_char",
        "insert": "insert_position",
        "delete": "delete_position",
        "remove": "delete_position",
        "set_car_plate": "set_plate",
        "update_plate": "set_plate",
        "get_state": "get_current_state",
        "detect_confusions": "detect_confusions_by_rules",
        "scan_confusions": "detect_confusions_by_rules",
        "remove_pending_confirmation": "remove_need_confirmation",
        "confirm_position": "add_confirmed",
    }
    return aliases.get(raw, raw)


class PlateToolExecutor:
    """执行模型规划出的工具调用，所有真实状态修改都集中在这里。"""

    def __init__(self, state: PlateAgentState) -> None:
        self.state = state
        self.plate_written_this_turn = False

    def execute_all(self, tool_calls: list[PlateToolCall]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for call in tool_calls:
            results.append(self.execute(call))
        return results

    def execute(self, call: PlateToolCall) -> dict[str, Any]:
        before_state = self.state.to_context()
        log_agent_line(
            "工具调用：准备执行",
            tool=call.name,
            参数=call.arguments,
            执行前状态=before_state,
        )
        try:
            success, message, data = self._execute(call)
        except Exception as exc:
            success = False
            message = f"工具执行异常：{exc}"
            data = {"error": str(exc)}
        if success and call.name in PLATE_WRITE_TOOL_NAMES:
            self.plate_written_this_turn = True

        after_state = self.state.to_context()
        result = {
            "tool_call_id": call.tool_call_id,
            "name": call.name,
            "arguments": call.arguments,
            "success": success,
            "message": message,
            "data": data,
            "before_state": before_state,
            "after_state": after_state,
        }
        log_agent_line(
            "工具调用：执行结果",
            tool=call.name,
            成功=success,
            结果=message,
            执行后状态=after_state,
        )
        log_session_event(
            "tool_call",
            tool_call_id=call.tool_call_id,
            tool=call.name,
            arguments=call.arguments,
            success=success,
            message=message,
            tool_result=data,
            before_state=before_state,
            after_state=after_state,
        )
        log_node_output("agent_tool.execute", result)
        return result

    def _execute(self, call: PlateToolCall) -> tuple[bool, str, dict[str, Any]]:
        name = call.name
        args = call.arguments
        if name == "get_current_state":
            return True, "已读取当前状态。", {"state": self.state.to_context()}
        if name == "validate_plate_rules":
            return self._validate_plate_rules(args)
        if name == "detect_confusions_by_rules":
            return self._detect_confusions_by_rules(args)
        if name == "set_plate":
            return self._set_plate(args)
        if name in EDIT_TOOL_NAMES:
            return self._edit_plate(name, args)
        if name in {"remove_need_confirmation", "add_confirmed", "remove_confirmed"}:
            return self._apply_confirmation_action(name, args)
        if name == "confirm_all":
            return self._confirm_all()
        return False, f"未知工具：{name}", {}

    def _set_plate(self, args: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        plate = normalize_candidate_plate(args.get("car_plate") or args.get("plate") or args.get("value"))
        if not plate:
            return False, "没有得到可用车牌内容。", {"attempted_plate": ""}
        if not is_valid_plate_number(plate):
            return False, "车牌格式不合法，状态未更新。", {
                "attempted_plate": plate,
                "plate_length": plate_length(plate),
                "vehicle_type": vehicle_type_by_length(plate),
                "valid": False,
            }

        confirmed_positions = parse_positions(args.get("confirmed_positions"), len(plate))
        self._write_plate(
            plate,
            confirmed_positions=confirmed_positions,
        )
        return True, "车牌状态已设置，并已按规则刷新二次确认列表。", {
            "car_plate": self.state.car_plate,
            "confirmed_positions": confirmed_positions,
            "confirmed": self.state.is_confirmed,
            "need_confirm_chars": [item.to_public_dict() for item in self.state.need_confirm_chars],
        }

    def _edit_plate(self, action: str, args: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        current_plate = clean_plate_text(self.state.car_plate)
        if not current_plate:
            return False, "当前没有暂存车牌，不能执行编辑。", {}
        command = PlateEditCommand(
            action=action,
            position=parse_positive_int(args.get("position") or args.get("index") or args.get("target_position")),
            value=normalize_edit_value(
                args.get("value")
                or args.get("new_value")
                or args.get("new_char")
                or args.get("char")
                or args.get("insert_value")
            ),
            old_value=normalize_edit_value(
                args.get("old_value")
                or args.get("old_char")
                or args.get("target_value")
                or args.get("target_char")
                or args.get("from")
            ),
            relation=normalize_relation(args.get("relation") or args.get("where")),
            occurrence=normalize_occurrence(args.get("occurrence") or args.get("which")),
            raw=args,
        )
        result = apply_plate_edit_command(current_plate, command)
        if not result.changed:
            return False, result.error or "编辑动作没有成功执行。", public_edit_result(result)
        if not is_valid_plate_number(result.car_plate):
            return False, "编辑后的车牌格式不合法，状态未更新。", public_edit_result(result)

        self._write_edited_plate(
            result.car_plate,
            previous_plate=current_plate,
            command=command,
            changed_positions=result.changed_positions,
        )
        data = public_edit_result(result)
        data["need_confirm_chars"] = [item.to_public_dict() for item in self.state.need_confirm_chars]
        return True, "车牌已按工具动作更新，并已更新二次确认列表。", data

    def _validate_plate_rules(self, args: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        plate = normalize_candidate_plate(args.get("car_plate") or args.get("plate") or self.state.car_plate)
        valid = is_valid_plate_number(plate)
        data = {
            "car_plate": plate,
            "valid": valid,
            "plate_length": plate_length(plate),
            "vehicle_type": vehicle_type_by_length(plate),
        }
        return True, "车牌规则校验完成。", data

    def _detect_confusions_by_rules(self, args: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        plate = normalize_candidate_plate(args.get("car_plate") or args.get("plate") or self.state.car_plate)
        if not plate:
            return False, "当前没有可扫描的车牌。", {}
        confusions = detect_initial_confusions_by_rule(plate)
        return True, "易混淆字符扫描完成，状态未自动更新。", {
            "car_plate": plate,
            "need_confirm_chars": [item.to_public_dict() for item in confusions],
        }

    def _refresh_confirmation_after_plate_write(
        self,
        *,
        pending_positions: set[int] | None = None,
        confirmed_positions: set[int] | None = None,
    ) -> None:
        plate = clean_plate_text(self.state.car_plate)
        if not plate:
            return
        confusions = detect_initial_confusions_by_rule(plate)
        rule_confusion_by_position = {item.position: item for item in confusions}
        requested_confirmed_set = {
            position
            for position in (confirmed_positions or set())
            if 1 <= position <= len(plate)
        }
        pending_set = {
            position
            for position in (pending_positions if pending_positions is not None else set(rule_confusion_by_position))
            if position in rule_confusion_by_position and position not in requested_confirmed_set
        }
        confirmed_set = {position for position in requested_confirmed_set if position not in pending_set}
        refresh_plate_state(
            self.state,
            plate,
            confusions=[rule_confusion_by_position[position] for position in sorted(pending_set)],
            confirmed=False,
            confirmed_positions=sorted(confirmed_set),
            preserve_confirmed=False,
        )

    def _apply_confirmation_action(self, action: str, args: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
        plate = clean_plate_text(self.state.car_plate)
        if not plate:
            return False, "当前没有暂存车牌，不能更新确认状态。", {}
        position = parse_positive_int(args.get("position") or args.get("index") or args.get("target_position"))
        if position <= 0 or position > len(plate):
            return False, "确认位置不在当前车牌范围内。", {"position": position, "plate_length": len(plate)}
        confirmation_action = PlateConfirmationAction(
            action=action,
            position=position,
            value=normalize_edit_value(args.get("value") or args.get("char")) or plate[position - 1],
            raw=args,
        )
        apply_confirmation_actions(self.state, [confirmation_action], source=f"tool.{action}")
        return True, "确认状态已更新。", {"action": confirmation_action.to_public_dict()}

    def _confirm_all(self) -> tuple[bool, str, dict[str, Any]]:
        if not self.state.car_plate:
            return False, "当前没有暂存车牌，不能确认。", {}
        if self.plate_written_this_turn:
            return False, "本轮已经写入或修改过车牌，不能直接确认整车牌。请先让用户确认修改后的新车牌，等待用户下一轮明确确认后再调用 confirm_all。", {
                "blocked_until_next_turn": True,
                "car_plate": self.state.car_plate,
            }
        apply_confirmation_actions(
            self.state,
            [PlateConfirmationAction(action="confirm_all")],
            source="tool.confirm_all",
        )
        self.state.final_car_plate = self.state.car_plate
        self.state.confirmed = True
        return True, "当前车牌已全部确认。", {"final_car_plate": self.state.final_car_plate}

    def _write_plate(
        self,
        plate: str,
        *,
        confirmed_positions: list[int],
    ) -> None:
        # 完整写入车牌时全量刷新规则待确认位。
        refresh_plate_state(
            self.state,
            plate,
            confusions=[],
            confirmed=False,
            preserve_confirmed=False,
        )
        self._refresh_confirmation_after_plate_write(
            confirmed_positions={position for position in confirmed_positions if position > 0}
        )

    def _write_edited_plate(
        self,
        plate: str,
        *,
        previous_plate: str,
        command: PlateEditCommand,
        changed_positions: list[int],
    ) -> None:
        pending_positions, confirmed_positions = self._preserved_confirmation_positions_after_edit(
            previous_plate=previous_plate,
            new_plate=plate,
            command=command,
            changed_positions=changed_positions,
        )
        refresh_plate_state(
            self.state,
            plate,
            confusions=[],
            confirmed=False,
            preserve_confirmed=False,
        )
        self._refresh_confirmation_after_plate_write(
            pending_positions=pending_positions,
            confirmed_positions=confirmed_positions,
        )

    def _preserved_confirmation_positions_after_edit(
        self,
        *,
        previous_plate: str,
        new_plate: str,
        command: PlateEditCommand,
        changed_positions: list[int],
    ) -> tuple[set[int], set[int]]:
        pending_positions: set[int] = set()
        confirmed_positions: set[int] = set()
        new_plate_text = clean_plate_text(new_plate)
        changed_set = {position for position in changed_positions if position > 0}

        def keep_status(old_position: int) -> int:
            if command.action in {"replace_position", "replace_char"}:
                return 0 if old_position in changed_set else old_position
            if command.action == "insert_position":
                insert_position = min(changed_set) if changed_set else 0
                if insert_position <= 0:
                    return old_position
                return old_position if old_position < insert_position else old_position + 1
            if command.action == "delete_position":
                delete_position = command.position
                if old_position <= 0 or delete_position <= 0 or old_position == delete_position:
                    return 0
                new_position = old_position if old_position < delete_position else old_position - 1
                return 0 if new_position == delete_position else new_position
            return old_position

        for item in self.state.need_confirm_chars:
            new_position = keep_status(item.position)
            if self._mapped_value_matches(previous_plate, new_plate_text, item.position, new_position):
                pending_positions.add(new_position)
        for item in self.state.confirmed_chars:
            new_position = keep_status(item.position)
            if self._mapped_value_matches(previous_plate, new_plate_text, item.position, new_position):
                confirmed_positions.add(new_position)

        if command.action in {"replace_position", "replace_char", "insert_position"}:
            confirmed_positions.update(position for position in changed_set if 1 <= position <= len(new_plate_text))
            pending_positions.difference_update(confirmed_positions)
        if command.action == "delete_position":
            delete_position = command.position
            pending_positions.discard(delete_position)
            confirmed_positions.discard(delete_position)
            if 1 <= delete_position <= len(new_plate_text):
                pending_positions.add(delete_position)

        return pending_positions, confirmed_positions

    @staticmethod
    def _mapped_value_matches(previous_plate: str, new_plate: str, old_position: int, new_position: int) -> bool:
        old_index = old_position - 1
        new_index = new_position - 1
        if old_index < 0 or old_index >= len(previous_plate) or new_index < 0 or new_index >= len(new_plate):
            return False
        return previous_plate[old_index] == new_plate[new_index]


def normalize_candidate_plate(value: Any) -> str:
    extracted = sanitize_extracted_plate_text(value)
    candidate = extracted or clean_plate_text(value)
    return normalize_plate_text(candidate)


def parse_positions(value: Any, max_position: int) -> list[int]:
    items = value if isinstance(value, list) else [value]
    positions = [parse_positive_int(item) for item in items]
    return unique_positions([position for position in positions if 1 <= position <= max_position])


def public_edit_result(result: PlateEditResult) -> dict[str, Any]:
    data: dict[str, Any] = {
        "car_plate": result.car_plate,
        "changed": result.changed,
        "changed_positions": result.changed_positions,
    }
    if result.command is not None:
        data["command"] = public_edit_command(result.command)
    if result.error:
        data["error"] = result.error
    if result.steps:
        data["steps"] = [public_edit_step(step) for step in result.steps]
    return data


def public_edit_step(step: dict[str, Any]) -> dict[str, Any]:
    command = step.get("command") if isinstance(step, dict) else None
    return {
        key: value
        for key, value in {
            "batch_index": step.get("batch_index") if isinstance(step, dict) else None,
            "input_plate": step.get("input_plate") if isinstance(step, dict) else None,
            "output_plate": step.get("output_plate") if isinstance(step, dict) else None,
            "changed": step.get("changed") if isinstance(step, dict) else None,
            "changed_positions": step.get("changed_positions") if isinstance(step, dict) else None,
            "error": step.get("error") if isinstance(step, dict) else None,
            "command": public_edit_command(command) if isinstance(command, dict) else command,
        }.items()
        if value not in (None, "", [])
    }


def public_edit_command(command: PlateEditCommand | dict[str, Any]) -> dict[str, Any]:
    if isinstance(command, PlateEditCommand):
        values = {
            "action": command.action,
            "position": command.position,
            "value": command.value,
            "old_value": command.old_value,
            "relation": command.relation,
            "occurrence": command.occurrence,
        }
    else:
        values = {
            "action": command.get("action"),
            "position": command.get("position"),
            "value": command.get("value"),
            "old_value": command.get("old_value"),
            "relation": command.get("relation"),
            "occurrence": command.get("occurrence"),
        }
    return {key: value for key, value in values.items() if value not in (None, "", [], 0)}


def parse_candidate_values(value: Any) -> list[str]:
    raw_items = value if isinstance(value, list) else []
    result: list[str] = []
    for item in raw_items:
        normalized = normalize_edit_value(item)
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def parse_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    raw = str(value or "").strip().lower()
    if raw in {"true", "yes", "1", "是", "对", "确认"}:
        return True
    if raw in {"false", "no", "0", "否", "不", "不确认"}:
        return False
    return default


def compact_observations(observations: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for item in observations[-limit:]:
        compacted.append(
            {
                "tool_call_id": item.get("tool_call_id"),
                "name": item.get("name"),
                "success": item.get("success"),
                "message": item.get("message"),
                "arguments": item.get("arguments"),
                "data": compact_result_data(item.get("data")),
                "after_state": compact_state(item.get("after_state")),
            }
        )
    return compacted


def compact_result_data(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    allowed = {
        "attempted_plate",
        "plate_length",
        "car_plate",
        "confirmed",
        "confirmed_positions",
        "changed",
        "changed_positions",
        "error",
        "final_car_plate",
        "need_confirm_chars",
        "blocked_until_next_turn",
        "valid",
        "vehicle_type",
    }
    return {key: value.get(key) for key in allowed if key in value}


def compact_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "car_plate",
        "plate_chars",
        "confirmed",
        "need_confirm_chars",
        "confirmed_chars",
        "vehicle_type",
        "final_car_plate",
    }
    return {key: value.get(key) for key in allowed if key in value}


def build_observation_text(observations: list[dict[str, Any]]) -> str:
    return (
        "<observations>\n"
        f"{json.dumps(compact_observations(observations), ensure_ascii=False, indent=2)}\n"
        "</observations>"
    )


def build_assistant_tool_call_message(plan: PlateAgentPlan) -> dict[str, Any]:
    """把模型 JSON 规划转成标准 assistant tool_calls 历史消息。"""

    return {
        "role": "assistant",
        "content": plan.thought or None,
        "tool_calls": [tool_call_to_protocol_dict(call) for call in plan.tool_calls],
    }


def tool_call_to_protocol_dict(call: PlateToolCall) -> dict[str, Any]:
    return {
        "id": call.tool_call_id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.arguments or {}, ensure_ascii=False),
        },
    }


def build_tool_observation_messages(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把后端工具返回值转成标准 tool role observation 历史消息。"""

    messages: list[dict[str, Any]] = []
    for observation in observations:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": str(observation.get("tool_call_id") or ""),
                "name": str(observation.get("name") or ""),
                "content": json.dumps(compact_observations([observation], limit=1)[0], ensure_ascii=False),
            }
        )
    return messages


def build_compatible_observation_message(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """兼容不支持 tool role 的模型服务，把 observation 降级成普通文本消息。"""

    return {
        "role": "user",
        "content": build_observation_text(observations),
    }
