from __future__ import annotations

import json
from typing import Any

from realtime_audio_demo.services.plate_agent_messages import build_edit_unclear_reply
from realtime_audio_demo.services.plate_agent_rules import clean_plate_text
from realtime_audio_demo.services.plate_agent_types import PlateAgentState


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


def reply_with_pending_confirmation(
    base_reply: str,
    state: PlateAgentState,
    *,
    include_confirmation: bool = True,
) -> str:
    return build_edit_unclear_reply(state, base_reply=base_reply, include_confirmation=include_confirmation)
