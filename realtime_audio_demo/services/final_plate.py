import json
import re

from realtime_audio_demo.services.interfaces import ChatModel
from realtime_audio_demo.services.output_filter import parse_assistant_json


async def resolve_confirmed_plate_output(
    *,
    model_client: ChatModel,
    model: str,
    assistant_text: str | None,
) -> str | None:
    data = parse_assistant_json(assistant_text)
    if not data:
        return assistant_text

    task_status = str(data.get("task_status") or "").strip().lower()
    if task_status != "confirmed":
        return assistant_text

    final_plate_number = clean_plate_text(data.get("normalized"))
    if not final_plate_number:
        return assistant_text

    data["final_plate_number"] = final_plate_number
    return json.dumps(data, ensure_ascii=False, indent=2)


def clean_plate_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or ""))
