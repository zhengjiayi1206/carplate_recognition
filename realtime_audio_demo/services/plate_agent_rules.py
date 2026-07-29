from __future__ import annotations

import re
from typing import Any

from realtime_audio_demo.services.plate_agent_types import PlateConfusion


PROVINCE_ABBREVIATIONS = {
    "京", "津", "冀", "晋", "蒙", "辽", "吉", "黑", "沪", "苏", "浙", "皖", "闽",
    "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "渝", "川", "贵", "云", "藏",
    "陕", "甘", "青", "宁", "新",
}
SPECIAL_PLATE_TAIL_CHARS = {"警", "临", "学", "领", "挂"}
CONFUSION_PROVINCE_CHARS = {"甘", "赣", "津", "京", "桂", "贵", "冀", "吉"}
CONFUSION_ALNUM_CHARS = {"2", "R", "1", "E"}


def clean_plate_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def normalize_plate_format(value: Any) -> str:
    return clean_plate_text(value).upper()


def first_char_is_ascii_letter_or_digit(value: str) -> bool:
    if not value:
        return False
    first = value[0]
    return first.isascii() and first.isalnum()


def replace_leading_g_with_ji(value: str) -> str:
    plate = normalize_plate_format(value)
    if plate.startswith("G"):
        return "冀" + plate[1:]
    return plate


def normalize_plate_text(value: Any) -> str:
    return replace_leading_g_with_ji(value)


def plate_length(car_plate: str) -> int:
    return len(clean_plate_text(car_plate))


def vehicle_type_by_length(car_plate: str) -> str:
    length = plate_length(car_plate)
    if length == 7:
        return "fuel"
    if length == 8:
        return "new_energy"
    return "unknown"


def is_valid_plate_number(car_plate: str) -> bool:
    plate = normalize_plate_text(car_plate)
    if vehicle_type_by_length(plate) == "unknown":
        return False
    if not plate or plate[0] not in PROVINCE_ABBREVIATIONS:
        return False
    if len(plate) < 2 or not re.fullmatch(r"[A-Z]", plate[1]):
        return False
    for index, char in enumerate(plate[2:], start=3):
        is_tail = index == len(plate)
        if char in SPECIAL_PLATE_TAIL_CHARS:
            if not is_tail:
                return False
            continue
        if not re.fullmatch(r"[A-Z0-9]", char):
            return False
    return True


def detect_initial_confusions_by_rule(car_plate: str) -> list[PlateConfusion]:
    plate = clean_plate_text(car_plate)
    confusions: list[PlateConfusion] = []
    if plate and plate[0] in CONFUSION_PROVINCE_CHARS:
        confusions.append(build_confusion(position=1, value=plate[0]))
    for index, value in enumerate(plate, start=1):
        if value in CONFUSION_ALNUM_CHARS:
            confusions.append(build_confusion(position=index, value=value))
    return with_relative_confusion_reasons(plate, confusions)


def is_rule_confusion_position(car_plate: str, position: int) -> bool:
    plate = clean_plate_text(car_plate)
    if position <= 0 or position > len(plate):
        return False
    value = plate[position - 1]
    return (position == 1 and value in CONFUSION_PROVINCE_CHARS) or value in CONFUSION_ALNUM_CHARS


def build_confusion(*, position: int, value: str) -> PlateConfusion:
    return PlateConfusion(
        position=position,
        value=value,
        reason=f"第{position}位当前识别为{describe_plate_char(value)}，请用户确认。",
    )


def with_relative_confusion_reasons(car_plate: str, confusions: list[PlateConfusion]) -> list[PlateConfusion]:
    plate = clean_plate_text(car_plate)
    normalized: list[PlateConfusion] = []
    for item in confusions:
        value = item.value
        index = item.position - 1
        if 0 <= index < len(plate):
            value = plate[index]
        candidates = list(item.candidates)
        if value and candidates and value not in candidates:
            candidates.insert(0, value)
        normalized.append(
            PlateConfusion(
                position=item.position,
                value=value,
                candidates=candidates,
                reason=relative_confusion_reason(plate, PlateConfusion(item.position, value, item.reason, candidates)),
            )
        )
    return normalized


def relative_confusion_reason(car_plate: str, item: PlateConfusion) -> str:
    plate = clean_plate_text(car_plate)
    index = item.position - 1
    value = item.value
    if 0 <= index < len(plate):
        value = value or plate[index]
    value_label = describe_plate_char(value)
    if 0 <= index < len(plate):
        return f"第{item.position}位当前识别为{value_label}，请用户确认是否正确。"
    return f"当前识别为{value_label}的位置，请用户确认是否正确。"


def describe_plate_char(value: str) -> str:
    value = str(value or "")
    labels = {
        "赣": "江西的赣",
        "甘": "甘肃的甘",
        "津": "天津的津",
        "京": "北京的京",
        "桂": "广西的桂",
        "贵": "贵州的贵",
        "冀": "河北的冀",
        "吉": "吉林的吉",
        "临": "临时车牌的临",
        "警": "警车的警",
        "学": "学车的学",
        "领": "领馆的领",
        "挂": "挂车的挂",
    }
    if value in labels:
        return labels[value]
    if value.isdigit():
        return f"数字 {value}"
    if re.fullmatch(r"[A-Za-z]", value):
        return f"字母 {value.upper()}"
    return value


def contains_absolute_position_text(value: str) -> bool:
    return bool(re.search(r"第\s*[0-9一二三四五六七八九十]+\s*位", str(value or "")))
