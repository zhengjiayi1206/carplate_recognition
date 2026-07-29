from __future__ import annotations

from typing import Any

from realtime_audio_demo.services.plate_agent_logging import log_agent_line, log_node_output
from realtime_audio_demo.services.plate_agent_rules import (
    is_rule_confusion_position,
    normalize_plate_text,
    vehicle_type_by_length,
    with_relative_confusion_reasons,
)
from realtime_audio_demo.services.plate_agent_types import PlateAgentState, PlateCharState, PlateConfusion, PlateEditResult


def clone_state(state: PlateAgentState) -> PlateAgentState:
    confusions: list[PlateConfusion] = []
    for item in state.confusions:
        cloned = PlateConfusion.from_value(item.to_dict())
        if cloned is not None:
            confusions.append(cloned)
    plate_chars = clone_plate_char_states(state.plate_chars)
    need_confirm_chars = clone_plate_char_states(state.need_confirm_chars)
    confirmed_chars = clone_plate_char_states(state.confirmed_chars)
    cloned_state = PlateAgentState(
        car_plate=state.car_plate,
        plate_chars=plate_chars,
        confirmed=state.confirmed,
        need_confirm_chars=need_confirm_chars,
        confirmed_chars=confirmed_chars,
        vehicle_type=state.vehicle_type,
        confusions=confusions,
        final_car_plate=state.final_car_plate,
        assistant_reply=state.assistant_reply,
        ack_sent=state.ack_sent,
        turn_summaries=list(state.turn_summaries),
    )
    if cloned_state.car_plate and not cloned_state.plate_chars:
        refresh_plate_state(
            cloned_state,
            cloned_state.car_plate,
            confusions=confusions,
            confirmed=cloned_state.is_confirmed,
            preserve_confirmed=False,
        )
    sanitize_need_confirmation_by_rules(cloned_state)
    return cloned_state


def clone_plate_char_states(items: list[PlateCharState]) -> list[PlateCharState]:
    cloned_items: list[PlateCharState] = []
    for item in items:
        cloned = PlateCharState.from_value(item.to_dict())
        if cloned is not None:
            cloned_items.append(cloned)
    return cloned_items


def refresh_plate_state(
    state: PlateAgentState,
    car_plate: str,
    *,
    confusions: list[PlateConfusion] | None = None,
    confirmed: bool = False,
    confirmed_positions: list[int] | None = None,
    preserve_confirmed: bool = True,
) -> None:
    before_state = state.to_context()
    plate = normalize_plate_text(car_plate)
    normalized_confusions = with_relative_confusion_reasons(plate, confusions or [])
    confirmed_position_set = {position for position in (confirmed_positions or []) if position > 0}
    previous_confirmed = collect_confirmed_char_keys(state) if preserve_confirmed else set()
    confusion_by_position = {item.position: item for item in normalized_confusions if item.position > 0}

    plate_chars: list[PlateCharState] = []
    for position, value in enumerate(plate, start=1):
        confusion = confusion_by_position.get(position)
        needs_confirmation = confusion is not None
        is_confirmed = bool(confirmed) or (
            not needs_confirmation
            and ((position, value) in previous_confirmed or position in confirmed_position_set)
        )
        plate_chars.append(
            PlateCharState(
                position=position,
                value=value,
                confirmed=is_confirmed,
                needs_confirmation=needs_confirmation,
                candidates=(confusion.candidates if confusion else []),
                reason=(confusion.reason if confusion else ""),
            )
        )

    state.car_plate = plate
    state.plate_chars = plate_chars
    state.vehicle_type = vehicle_type_by_length(plate)
    state.confusions = normalized_confusions
    state.confirmed = bool(confirmed)
    if state.confirmed:
        state.final_car_plate = plate
    else:
        state.final_car_plate = ""
    state.need_confirm_chars = [item for item in plate_chars if item.needs_confirmation]
    state.confirmed_chars = [item for item in plate_chars if item.confirmed]
    log_agent_line(
        "更新二次确认列表",
        当前车牌=state.car_plate,
        二次确认列表=format_plate_char_states(state.need_confirm_chars),
        说明="这些位置还需要继续向用户确认。",
    )
    log_agent_line(
        "更新已经完成确认字符",
        当前车牌=state.car_plate,
        已确认字符=format_plate_char_states(state.confirmed_chars),
        说明="这些位置已经由用户确认或本轮修改后确认。",
    )
    log_node_output(
        "refresh_plate_state",
        {
            "action": "refresh_plate_state",
            "input": {
                "car_plate": car_plate,
                "normalized_car_plate": plate,
                "confirmed": confirmed,
                "confirmed_positions": confirmed_positions or [],
                "preserve_confirmed": preserve_confirmed,
                "confusions": [item.to_dict() for item in (confusions or [])],
            },
            "before_state": before_state,
            "state": state.to_context(),
        },
    )


def sanitize_need_confirmation_by_rules(state: PlateAgentState) -> None:
    plate = normalize_plate_text(state.car_plate)
    if not plate:
        return
    confirmed_positions = {item.position for item in state.confirmed_chars if item.position > 0}
    filtered_by_position: dict[int, PlateConfusion] = {}
    for item in state.confusions:
        if item.position in confirmed_positions:
            continue
        if is_rule_confusion_position(plate, item.position):
            filtered_by_position[item.position] = item
    for item in state.need_confirm_chars:
        if item.position in confirmed_positions:
            continue
        if item.position not in filtered_by_position and is_rule_confusion_position(plate, item.position):
            filtered_by_position[item.position] = PlateConfusion(
                position=item.position,
                value=item.value,
                reason=item.reason,
                candidates=list(item.candidates),
            )
    expected_positions = set(filtered_by_position)
    current_positions = {item.position for item in state.need_confirm_chars if item.position > 0}
    if expected_positions == current_positions and all(
        is_rule_confusion_position(plate, item.position)
        for item in state.need_confirm_chars
    ):
        return
    refresh_plate_state(
        state,
        plate,
        confusions=list(filtered_by_position.values()),
        confirmed=state.is_confirmed,
        confirmed_positions=sorted(confirmed_positions),
        preserve_confirmed=False,
    )


def collect_confirmed_char_keys(state: PlateAgentState) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    for item in [*state.confirmed_chars, *state.plate_chars]:
        if item.confirmed and item.position > 0 and item.value:
            keys.add((item.position, item.value))
    return keys


def extract_batch_commands(edit_result: PlateEditResult) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for step in edit_result.steps:
        command = step.get("command") if isinstance(step, dict) else None
        if isinstance(command, dict):
            commands.append(command)
    if commands:
        return commands
    return [edit_result.command.to_dict()] if edit_result.command else []


def format_plate_char_states(items: list[PlateCharState]) -> list[str]:
    result: list[str] = []
    for item in items:
        label = f"第{item.position}位={item.value}"
        if item.reason:
            label += f"，原因：{item.reason}"
        result.append(label)
    return result
