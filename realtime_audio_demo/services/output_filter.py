import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class AssistantOutput:
    raw_text: str
    display_text: str
    history_text: str
    speech_text: str
    is_json: bool


def normalize_assistant_output(text: str | None) -> AssistantOutput:
    raw_text = (text or "").strip()
    if not raw_text:
        return AssistantOutput(
            raw_text="",
            display_text="",
            history_text="",
            speech_text="",
            is_json=False,
        )

    candidate = extract_json_candidate(raw_text)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return AssistantOutput(
            raw_text=raw_text,
            display_text=raw_text,
            history_text=raw_text,
            speech_text=raw_text,
            is_json=False,
        )

    if not isinstance(value, dict):
        display_text = format_json_value(value)
        return AssistantOutput(
            raw_text=raw_text,
            display_text=display_text,
            history_text=display_text,
            speech_text="",
            is_json=True,
        )

    display_text = format_json_value(value)

    return AssistantOutput(
        raw_text=raw_text,
        display_text=display_text,
        history_text=display_text,
        speech_text=display_text,
        is_json=True,
    )


def extract_json_candidate(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json|JSON)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return stripped


def parse_assistant_json(text: str | None) -> dict[str, Any] | None:
    candidate = extract_json_candidate((text or "").strip())
    if not candidate:
        return None
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def validate_current_normalized_plate(text: str | None) -> str | None:
    data = parse_assistant_json(text)
    if not data or "normalized" not in data:
        return text

    normalized = re.sub(r"\s+", "", str(data.get("normalized") or ""))
    data["normalized"] = normalized
    plate_length = len(normalized)
    if plate_length in {7, 8}:
        return format_json_value(data)

    if plate_length < 7:
        data["task_status"] = "need_more_info"
        data["assistant_reply"] = "当前您提供的车牌位数不符合要求。请您重新输入车牌。"
    else:
        data["task_status"] = "invalid"
        data["assistant_reply"] = "请重新输入车牌，车牌格式不符合要求。"

    return format_json_value(data)


def format_json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
