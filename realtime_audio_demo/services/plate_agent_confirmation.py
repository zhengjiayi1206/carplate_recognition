from __future__ import annotations

from typing import Any

from realtime_audio_demo.services.plate_agent_edit import normalize_edit_value, parse_positive_int, parse_json_value
from realtime_audio_demo.services.plate_agent_logging import log_agent_line, log_node_output
from realtime_audio_demo.services.plate_agent_rules import (
    clean_plate_text,
    describe_plate_char,
    is_rule_confusion_position,
    with_relative_confusion_reasons,
)
from realtime_audio_demo.services.plate_agent_state import collect_confirmed_char_keys, refresh_plate_state
from realtime_audio_demo.services.plate_agent_types import PlateAgentState, PlateConfirmationAction, PlateConfusion


CONFIRMATION_ACTIONS = {
    "add_need_confirmation",
    "remove_need_confirmation",
    "clear_need_confirmation",
    "add_confirmed",
    "remove_confirmed",
    "clear_confirmed",
    "confirm_all",
    "none",
}


def parse_confirmation_actions(text: Any) -> list[PlateConfirmationAction]:
    value = parse_json_value(text)
    items = extract_confirmation_action_items(value)
    actions = [parse_confirmation_action_data(item) for item in items if isinstance(item, dict)]
    return actions or [PlateConfirmationAction(action="none", raw={})]


def extract_confirmation_action_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return []

    for key in ("actions", "state_actions", "confirmation_actions"):
        items = value.get(key)
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]

    return [value]


def parse_confirmation_action_data(data: dict[str, Any]) -> PlateConfirmationAction:
    return PlateConfirmationAction(
        action=normalize_confirmation_action(data.get("action") or data.get("name")),
        position=parse_positive_int(data.get("position") or data.get("index") or data.get("target_position")),
        value=normalize_edit_value(data.get("value") or data.get("char")),
        candidates=parse_candidate_values(data.get("candidates")),
        reason=str(data.get("reason") or "").strip(),
        raw=data,
    )


def normalize_confirmation_action(value: Any) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "add_need_confirm": "add_need_confirmation",
        "add_need_confirmation": "add_need_confirmation",
        "add_pending": "add_need_confirmation",
        "need_confirm": "add_need_confirmation",
        "remove_need_confirm": "remove_need_confirmation",
        "remove_need_confirmation": "remove_need_confirmation",
        "remove_pending": "remove_need_confirmation",
        "clear_need_confirm": "clear_need_confirmation",
        "clear_need_confirmation": "clear_need_confirmation",
        "clear_pending": "clear_need_confirmation",
        "add_confirm": "add_confirmed",
        "add_confirmed": "add_confirmed",
        "confirm_position": "add_confirmed",
        "remove_confirm": "remove_confirmed",
        "remove_confirmed": "remove_confirmed",
        "clear_confirm": "clear_confirmed",
        "clear_confirmed": "clear_confirmed",
        "confirm_all": "confirm_all",
        "none": "none",
    }
    return aliases.get(raw, raw if raw in CONFIRMATION_ACTIONS else "none")


def parse_candidate_values(value: Any) -> list[str]:
    raw_items = value if isinstance(value, list) else []
    candidates: list[str] = []
    for item in raw_items:
        candidate = normalize_edit_value(item)
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def confirmation_actions_from_confusions(
    confusions: list[PlateConfusion],
    *,
    confirmed_positions: list[int] | None = None,
) -> list[PlateConfirmationAction]:
    confirmed_set = {position for position in (confirmed_positions or []) if position > 0}
    actions: list[PlateConfirmationAction] = [PlateConfirmationAction(action="clear_need_confirmation")]
    for position in sorted(confirmed_set):
        actions.append(PlateConfirmationAction(action="add_confirmed", position=position))
    for item in confusions:
        if item.position in confirmed_set:
            continue
        actions.append(
            PlateConfirmationAction(
                action="add_need_confirmation",
                position=item.position,
                value=item.value,
                candidates=list(item.candidates),
                reason=item.reason,
            )
        )
    return actions


def complete_confirmation_actions(
    *,
    plate: str,
    rule_confusions: list[PlateConfusion],
    model_actions: list[PlateConfirmationAction],
    confirmed_positions: list[int] | None = None,
) -> list[PlateConfirmationAction]:
    actions = [PlateConfirmationAction(action="clear_need_confirmation")]
    actions.extend(PlateConfirmationAction(action="add_confirmed", position=position) for position in confirmed_positions or [])
    actions.extend(action for action in model_actions if action.action != "clear_need_confirmation")

    settled_positions = {
        action.position
        for action in actions
        if action.position > 0 and action.action in {"add_confirmed", "remove_need_confirmation"}
    }
    need_positions = {
        action.position
        for action in actions
        if action.position > 0 and action.action == "add_need_confirmation"
    }
    for item in rule_confusions:
        if item.position in settled_positions or item.position in need_positions:
            continue
        actions.append(
            PlateConfirmationAction(
                action="add_need_confirmation",
                position=item.position,
                value=item.value,
                candidates=list(item.candidates),
                reason=item.reason,
            )
        )
    return normalize_confirmation_actions_for_plate(plate, actions)


def normalize_confirmation_actions_for_plate(
    plate: str,
    actions: list[PlateConfirmationAction],
) -> list[PlateConfirmationAction]:
    plate_text = clean_plate_text(plate)
    normalized: list[PlateConfirmationAction] = []
    for action in actions:
        if action.action in {"clear_need_confirmation", "clear_confirmed", "confirm_all", "none"}:
            normalized.append(action)
            continue
        if action.position <= 0 or action.position > len(plate_text):
            continue
        value = action.value or plate_text[action.position - 1]
        normalized.append(
            PlateConfirmationAction(
                action=action.action,
                position=action.position,
                value=value,
                candidates=list(action.candidates),
                reason=action.reason,
                raw=action.raw,
            )
        )
    return normalized or [PlateConfirmationAction(action="none")]


def apply_confirmation_actions(
    state: PlateAgentState,
    actions: list[PlateConfirmationAction],
    *,
    source: str,
) -> list[PlateConfusion]:
    before_state = state.to_context()
    plate = clean_plate_text(state.car_plate)
    current_confusions = {
        item.position: item
        for item in with_relative_confusion_reasons(plate, state.confusions)
        if item.position > 0 and is_rule_confusion_position(plate, item.position)
    }
    confirmed_positions = {position for position, _ in collect_confirmed_char_keys(state)}
    confirm_all = False
    normalized_actions = normalize_confirmation_actions_for_plate(plate, actions)

    log_agent_line(
        "确认状态更新：action 是什么",
        来源=source,
        当前车牌=plate,
        actions=[item.to_dict() for item in normalized_actions],
    )

    for action in normalized_actions:
        if action.action == "none":
            continue
        if action.action == "clear_need_confirmation":
            current_confusions.clear()
            continue
        if action.action == "clear_confirmed":
            confirmed_positions.clear()
            continue
        if action.action == "confirm_all":
            current_confusions.clear()
            confirmed_positions = set(range(1, len(plate) + 1))
            confirm_all = True
            continue
        if action.position <= 0 or action.position > len(plate):
            continue
        if action.action == "add_need_confirmation":
            if not is_rule_confusion_position(plate, action.position):
                continue
            value = plate[action.position - 1]
            current_confusions[action.position] = PlateConfusion(
                position=action.position,
                value=value,
                candidates=list(action.candidates),
                reason=action.reason or f"第{action.position}位当前识别为{describe_plate_char(value)}，请用户确认。",
            )
            confirmed_positions.discard(action.position)
            continue
        if action.action == "remove_need_confirmation":
            current_confusions.pop(action.position, None)
            continue
        if action.action == "add_confirmed":
            current_confusions.pop(action.position, None)
            confirmed_positions.add(action.position)
            continue
        if action.action == "remove_confirmed":
            confirmed_positions.discard(action.position)

    confusions = with_relative_confusion_reasons(plate, list(current_confusions.values()))
    refresh_plate_state(
        state,
        plate,
        confusions=confusions,
        confirmed=confirm_all,
        confirmed_positions=sorted(confirmed_positions),
        preserve_confirmed=False,
    )
    log_agent_line(
        "确认状态更新：action 执行结果",
        来源=source,
        当前车牌=state.car_plate,
        二次确认列表=[item.to_dict() for item in state.need_confirm_chars],
        已确认字符=[item.to_dict() for item in state.confirmed_chars],
    )
    log_node_output(
        "confirmation_state.apply_actions",
        {
            "source": source,
            "actions": [item.to_dict() for item in normalized_actions],
            "before_state": before_state,
            "state": state.to_context(),
        },
    )
    return confusions
