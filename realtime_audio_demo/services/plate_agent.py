from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from realtime_audio_demo.config import QWEN_MODEL
from realtime_audio_demo.services.interfaces import ChatModel
from realtime_audio_demo.services.output_filter import extract_json_candidate


logger = logging.getLogger("uvicorn.error")

NO_PLATE_REPLY = "我没有听到车牌号内容，请告诉我车牌号。"
INVALID_PLATE_REPLY = "您好，您当前的车牌号并不是有效号码，请重新输入。"

CAR_PLATE_EXTRACTION_PROMPT_PATH = Path(__file__).resolve().parents[1] / "car_plate_extraction_prompt.md"
CAR_PLATE_EXTRACTION_PROMPT = CAR_PLATE_EXTRACTION_PROMPT_PATH.read_text(encoding="utf-8").strip()
PRONUNCIATION_RULES_PROMPT = """
### pronunciation 重点规则

识别车牌时必须同时注意用户报号发音。用户说的是车牌字符的读音，不要把读音汉字写进 car_plate。

| 标准字符 | 用户音频里可能说成 | 说明 |
|----------|--------------------|------|
| `0` | 洞 | 报号码时常用“洞”表示数字 0 |
| `1` | 幺 | 报号码时常用“幺”表示数字 1 |
| `4` | 是 | 连续报车牌时，“是”可能是数字 4 的近音 |
| `6` | 陆 | “陆”是数字 6 在报号场景中的常用读法 |
| `7` | 拐 | 报号码时常用“拐”表示数字 7 |
| `C` | 吸 | 用户发音不标准时，C 可能听起来像“吸” |
| `J` | 勾、沟儿 | 用户报车牌时可能用“勾”或“沟儿”表示字母 J |
| `Q` | 圈 | 用户报车牌时可能用“圈”表示字母 Q |

car_plate 只能包含省份简称、英文字母、数字，不能包含洞、幺、是、陆、拐、吸、勾、沟儿、圈。
"""


def log_node_output(node: str, output: dict[str, Any]) -> None:
    logger.info("plate_agent node=%s output=%s", node, json.dumps(output, ensure_ascii=False, default=str))


@dataclass(slots=True)
class PlateConfusion:
    position: int
    value: str
    reason: str
    candidates: list[str] = field(default_factory=list)

    @classmethod
    def from_value(cls, value: Any) -> "PlateConfusion | None":
        if not isinstance(value, dict):
            return None
        try:
            position = int(value.get("position") or 0)
        except (TypeError, ValueError):
            position = 0
        text_value = str(value.get("value") or "").strip()
        reason = str(value.get("reason") or "").strip()
        candidates_raw = value.get("candidates")
        candidates = [str(item).strip() for item in candidates_raw] if isinstance(candidates_raw, list) else []
        if position <= 0 and not text_value and not reason:
            return None
        return cls(position=position, value=text_value, reason=reason, candidates=[item for item in candidates if item])

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "value": self.value,
            "candidates": self.candidates,
            "reason": self.reason,
        }


@dataclass(slots=True)
class PlateAgentState:
    car_plate: str = ""
    vehicle_type: str = "unknown"
    confusions: list[PlateConfusion] = field(default_factory=list)
    final_car_plate: str = ""
    assistant_reply: str = ""

    @property
    def has_car_plate(self) -> bool:
        return bool(self.car_plate)

    @property
    def is_confirmed(self) -> bool:
        return bool(self.final_car_plate)

    def to_context(self) -> dict[str, Any]:
        return {
            "car_plate": self.car_plate,
            "vehicle_type": self.vehicle_type,
            "confusions": [item.to_dict() for item in self.confusions],
            "final_car_plate": self.final_car_plate,
        }


@dataclass(slots=True)
class PlateAgentResult:
    text: str
    history_text: str
    speech_text: str
    state: PlateAgentState
    latency_ms: int
    debug: dict[str, Any] = field(default_factory=dict)


class PlateAgentService:
    def __init__(self, model_client: ChatModel) -> None:
        self.model_client = model_client

    async def handle_audio_turn(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        state: PlateAgentState,
    ) -> PlateAgentResult:
        started = time.perf_counter()
        debug: dict[str, Any] = {}
        working = clone_state(state)

        if not working.has_car_plate:
            has_plate = await self.detect_plate_presence(model=model, wav_bytes=wav_bytes)
            debug["has_plate"] = has_plate
            if not has_plate:
                output = build_output_json(
                    task_status="need_more_info",
                    car_plate="",
                    assistant_reply=NO_PLATE_REPLY,
                )
                working.assistant_reply = NO_PLATE_REPLY
                latency_ms = elapsed_ms(started)
                log_node_output(
                    "turn_result",
                    {
                        "stage": "no_plate",
                        "text": output,
                        "speech_text": NO_PLATE_REPLY,
                        "state": working.to_context(),
                        "latency_ms": latency_ms,
                    },
                )
                return PlateAgentResult(
                    text=output,
                    history_text=output,
                    speech_text=NO_PLATE_REPLY,
                    state=working,
                    latency_ms=latency_ms,
                    debug=debug,
                )

            car_plate = await self.extract_car_plate(model=model, wav_bytes=wav_bytes)
            vehicle_type = vehicle_type_by_length(car_plate)
            if vehicle_type == "unknown":
                return self.build_invalid_plate_result(
                    started=started,
                    working=working,
                    car_plate=car_plate,
                    debug=debug,
                    stage="invalid_initial_plate",
                )
            working.car_plate = car_plate
            working.final_car_plate = ""
            working.vehicle_type = vehicle_type
            log_node_output(
                "resolve_vehicle_type_by_length",
                {
                    "car_plate": working.car_plate,
                    "plate_length": plate_length(working.car_plate),
                    "vehicle_type": working.vehicle_type,
                },
            )
            confusions = detect_initial_confusions_by_rule(working.car_plate)
            log_node_output(
                "detect_confusions",
                {
                    "source": "rule",
                    "car_plate": working.car_plate,
                    "confusions": [item.to_dict() for item in confusions],
                },
            )
            working.confusions = confusions
            assistant_reply = await self.generate_reply(model=model, state=working, changed=True)
            working.assistant_reply = assistant_reply
            output = build_output_json(
                task_status="need_confirmation",
                car_plate=working.car_plate,
                assistant_reply=assistant_reply,
            )
            latency_ms = elapsed_ms(started)
            result_debug = {
                **debug,
                "car_plate": working.car_plate,
                "vehicle_type": working.vehicle_type,
                "confusions": [item.to_dict() for item in working.confusions],
            }
            log_node_output(
                "turn_result",
                {
                    "stage": "initial_plate",
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
                debug=result_debug,
            )

        confirmation = await self.detect_confirmation(model=model, wav_bytes=wav_bytes, state=working)
        debug["confirmation"] = confirmation
        if confirmation:
            working.final_car_plate = working.car_plate
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

        new_car_plate = await self.update_car_plate(model=model, wav_bytes=wav_bytes, state=working)
        if new_car_plate:
            vehicle_type = vehicle_type_by_length(new_car_plate)
            if vehicle_type == "unknown":
                return self.build_invalid_plate_result(
                    started=started,
                    working=working,
                    car_plate=new_car_plate,
                    debug=debug,
                    stage="invalid_updated_plate",
                )
            working.car_plate = new_car_plate
            working.final_car_plate = ""
            working.vehicle_type = vehicle_type
            log_node_output(
                "resolve_vehicle_type_by_length",
                {
                    "car_plate": working.car_plate,
                    "plate_length": plate_length(working.car_plate),
                    "vehicle_type": working.vehicle_type,
                },
            )
        rule_confusions = detect_initial_confusions_by_rule(working.car_plate)
        log_node_output(
            "detect_confusions.rule_scan",
            {
                "source": "rule_before_model",
                "car_plate": working.car_plate,
                "confusions": [item.to_dict() for item in rule_confusions],
            },
        )
        confusions = await self.detect_confusions(
            model=model,
            wav_bytes=wav_bytes,
            car_plate=working.car_plate,
            state=working,
            rule_confusions=rule_confusions,
        )
        working.confusions = confusions
        assistant_reply = await self.generate_reply(model=model, state=working, changed=True)
        working.assistant_reply = assistant_reply
        output = build_output_json(
            task_status="need_confirmation",
            car_plate=working.car_plate,
            assistant_reply=assistant_reply,
        )
        latency_ms = elapsed_ms(started)
        result_debug = {
            **debug,
            "car_plate": working.car_plate,
            "vehicle_type": working.vehicle_type,
            "confusions": [item.to_dict() for item in working.confusions],
        }
        log_node_output(
            "turn_result",
            {
                "stage": "updated_plate",
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
            debug=result_debug,
        )

    def build_invalid_plate_result(
        self,
        *,
        started: float,
        working: PlateAgentState,
        car_plate: str,
        debug: dict[str, Any],
        stage: str,
    ) -> PlateAgentResult:
        output = build_output_json(
            task_status="invalid",
            car_plate=car_plate,
            assistant_reply=INVALID_PLATE_REPLY,
        )
        working.car_plate = ""
        working.vehicle_type = "unknown"
        working.confusions = []
        working.final_car_plate = ""
        working.assistant_reply = INVALID_PLATE_REPLY
        latency_ms = elapsed_ms(started)
        result_debug = {
            **debug,
            "invalid_car_plate": clean_plate_text(car_plate),
            "plate_length": plate_length(car_plate),
        }
        log_node_output(
            "validate_plate_length",
            {
                "car_plate": clean_plate_text(car_plate),
                "plate_length": plate_length(car_plate),
                "valid": False,
                "assistant_reply": INVALID_PLATE_REPLY,
            },
        )
        log_node_output(
            "turn_result",
            {
                "stage": stage,
                "text": output,
                "speech_text": INVALID_PLATE_REPLY,
                "state": working.to_context(),
                "latency_ms": latency_ms,
            },
        )
        return PlateAgentResult(
            text=output,
            history_text=output,
            speech_text=INVALID_PLATE_REPLY,
            state=working,
            latency_ms=latency_ms,
            debug=result_debug,
        )

    async def detect_plate_presence(self, *, model: str, wav_bytes: bytes) -> bool:
        result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=(
                "任务：判断用户语音中是否包含车牌号相关内容。"
                "只回答 true 或 false，不要输出其它内容。"
                "如果用户说了省份、字母、数字、车牌片段或完整车牌，回答 true。"
            ),
            max_tokens=8,
        )
        has_plate = parse_bool_text(result, default=False)
        log_node_output("detect_plate_presence", {"raw": result, "has_plate": has_plate})
        return has_plate

    async def extract_car_plate(self, *, model: str, wav_bytes: bytes) -> str:
        extraction_result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=f"{CAR_PLATE_EXTRACTION_PROMPT}\n\n{PRONUNCIATION_RULES_PROMPT}",
            max_tokens=128,
        )
        extraction_plate = clean_plate_text(parse_json_object(extraction_result).get("car_plate"))
        log_node_output("extract_car_plate.step1_extract_with_pronunciation", {"raw": extraction_result, "car_plate": extraction_plate})

        suffix_result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=f"""
任务：复核用户音频中的车牌末尾是否包含特殊车牌尾字，并修正第一步识别结果。只输出 JSON 对象，字段为 car_plate。
第一步识别结果：{extraction_plate}

这一步只重点检查车牌最后一位。用户的车牌末尾可能不是数字或字母，而是特殊车牌尾字。

### 特殊车牌尾字

| 尾字 | 含义 |
|------|------|
| 警 | 警车 |
| 临 | 临时车牌 |
| 学 | 教练车 |
| 领 | 领事馆车牌 |
| 挂 | 半挂车牌 |

如果音频里明确说了这些特殊尾字，car_plate 末尾要保留对应汉字。
如果音频里没有明确特殊尾字，就保持第一步识别结果。

### 临字重点规则

“临”是临时车牌尾字，语音里可能被听成“零”“0”“洞”“林”“拎”“令”。
如果用户是在车牌末尾说“临”“临牌”“临时牌”“临时车牌”，或者末尾发音明显接近“临”，必须把 car_plate 最后一位改成“临”，不要输出 0。
如果用户明确是在报数字“零”或“洞”，并且不是临时车牌语义，才保留数字 0。

### 示例

第一步识别结果：粤B12340
音频末尾说的是：粤 B 一二三四 临
输出：{{"car_plate":"粤B1234临"}}

第一步识别结果：京A88880
音频末尾说的是：京 A 八八八八 临牌
输出：{{"car_plate":"京A8888临"}}

第一步识别结果：沪C12340
音频末尾说的是：沪 C 一二三四 洞
输出：{{"car_plate":"沪C12340"}}
""",
            max_tokens=128,
        )
        suffix_plate = clean_plate_text(parse_json_object(suffix_result).get("car_plate")) or extraction_plate
        final_plate = await self.normalize_plate_result(
            model=model,
            wav_bytes=wav_bytes,
            car_plate=suffix_plate,
            node="extract_car_plate.normalize",
        )
        log_node_output(
            "extract_car_plate.step3_special_suffix",
            {"raw": suffix_result, "previous_car_plate": extraction_plate, "car_plate": final_plate},
        )
        log_node_output(
            "extract_car_plate",
            {
                "step1_car_plate": extraction_plate,
                "step3_car_plate": suffix_plate,
                "car_plate": final_plate,
            },
        )
        return final_plate

    async def detect_confusions(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        car_plate: str,
        state: PlateAgentState | None = None,
        rule_confusions: list[PlateConfusion] | None = None,
    ) -> list[PlateConfusion]:
        current_context = {
            "car_plate": clean_plate_text(car_plate),
            "vehicle_type": (state.vehicle_type if state else vehicle_type_by_length(car_plate)),
            "assistant_reply": state.assistant_reply if state else "",
        }
        rule_confusions_context = json.dumps(
            [item.to_dict() for item in (rule_confusions or [])],
            ensure_ascii=False,
        )
        result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=(
                "任务：结合用户最新音频、当前识别车牌、当前候选混淆列表，判断哪些混淆点还需要二次确认。"
                f"当前识别车牌：{car_plate}。"
                f"当前识别状态：{json.dumps(current_context, ensure_ascii=False)}。"
                f"当前候选混淆列表：{rule_confusions_context}。"
                "当前候选混淆列表来自固定规则扫描，表示当前新车牌里命中的潜在混淆字符。"
                "不要参考上一轮旧的 confusions；不要把上一轮旧混淆点带入本轮结果。"
                "处理顺序必须如下："
                "第一步，只从当前候选混淆列表里判断哪些还需要用户二次确认。"
                "如果用户最新音频已经明确确认了某个候选混淆点，就不要输出这个混淆点。"
                "如果用户最新音频没有明确确认某个候选混淆点，或者音频里仍然听起来不确定，就输出这个混淆点。"
                "第二步，如果用户最新音频里还有其他明显不确定、发音含糊、背景噪声导致可能听错的位置，可以额外输出。"
                "只要输出混淆点，每一个字符位置都必须单独输出一条 confusions 记录。"
                "不能只返回第一个命中的位置，不能合并多个位置。"
                "当前候选混淆列表的规则来源如下："
                "1. 如果某个字符是 2 或 R，必须输出这个字符的 position，candidates 必须包含 2 和 R，reason 使用前面的、后面的、靠前的、靠后的等相对说法，不能说第几位；"
                "2. 如果某个字符是 1 或 E，必须输出这个字符的 position，candidates 必须包含 1 和 E，reason 使用前面的、后面的、靠前的、靠后的等相对说法，不能说第几位；"
                "3. 如果省份简称是 甘 或 赣，必须输出 position=1，reason 说明车牌开头的省份简称当前识别为甘肃的甘或江西的赣，不能说第几位；"
                "4. 如果省份简称是 津 或 京，必须输出 position=1，candidates 必须包含 津 和 京，reason 说明车牌开头的省份简称当前识别为天津的津或北京的京，不能说第几位；"
                "5. 如果省份简称是 桂 或 贵，必须输出 position=1，reason 说明车牌开头的省份简称当前识别为广西的桂或贵州的贵，不能说第几位；"
                "6. 如果省份简称是 冀 或 吉，必须输出 position=1，reason 说明车牌开头的省份简称当前识别为河北的冀或吉林的吉，不能说第几位；"
                "7. 其他语音中明显不确定、用户发音含糊、背景噪声导致可能听错的位置，也可以额外标记出来。"
                "只输出 JSON 对象，字段为 confusions。confusions 是数组，每一项包含 position、value、candidates、reason。"
                "position 是给后端使用的数字位置；reason 是给用户确认的自然语言，不能出现“第几位”“第1位”“第2位”等绝对位置表达。"
                "如果没有需要二次确认的易混淆字符，confusions 输出空数组。"
            ),
            max_tokens=256,
        )
        data = parse_json_object(result)
        raw_items = data.get("confusions")
        if not isinstance(raw_items, list):
            log_node_output("detect_confusions", {"raw": result, "car_plate": car_plate, "confusions": []})
            return []
        items = [PlateConfusion.from_value(item) for item in raw_items]
        confusions = with_relative_confusion_reasons(car_plate, [item for item in items if item is not None])
        log_node_output(
            "detect_confusions",
            {
                "raw": result,
                "car_plate": car_plate,
                "rule_confusions": [item.to_dict() for item in (rule_confusions or [])],
                "confusions": [item.to_dict() for item in confusions],
            },
        )
        return confusions

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
            max_tokens=8,
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

    async def update_car_plate(self, *, model: str, wav_bytes: bytes, state: PlateAgentState) -> str:
        update_context = json.dumps(
            {
                **state.to_context(),
                "assistant_reply": state.assistant_reply,
            },
            ensure_ascii=False,
        )
        base_result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=f"""{CAR_PLATE_EXTRACTION_PROMPT}

{PRONUNCIATION_RULES_PROMPT}

你正在更新一个已经识别过的车牌号，不是开始新的车牌识别任务。

任务：当前车牌可能存在问题，请一定根据这一轮用户音频里的要求，对当前暂时识别到的车牌进行更改。

判断规则：
1. 如果用户明确说某个位置、某个字符、前面的、后面的、开头、末尾不对，必须按用户这次音频修改当前车牌。
2. 如果用户只是修正一个字符，只替换这个字符，并保留其他已经识别好的字符。
3. 如果用户补充缺失字符，把补充内容合并到当前车牌中，输出完整的新 car_plate。
4. 如果用户重新报了完整车牌，以用户这次音频为准输出完整的新 car_plate。
5. 如果用户只是确认上一轮 AI 询问的某个混淆点正确，不要把车牌改错，保持当前 car_plate。
6. 如果用户这次没有提供任何可用于修改车牌的新信息，输出当前暂时识别到的车牌。
7. 必须保持车牌长度不变。除非用户明确说了"删除"或"去掉"这样的意思，否则不能减少车牌号中的字符数量。如果用户说某个字符不对，应该用正确字符替换它，而不是直接删掉。

输出要求：
只输出 JSON 对象，不要解释，不要 Markdown，不要输出用户原话。
格式必须是：
{{"car_plate":"..."}}

当前暂时识别到的车牌：{state.car_plate}
上一轮 AI 回复给用户的内容：{state.assistant_reply}
这一轮用户输入：本次请求随带的用户音频。
""",
            max_tokens=128,
        )
        base_plate = clean_plate_text(parse_json_object(base_result).get("car_plate")) or state.car_plate
        log_node_output(
            "update_car_plate.step1_update_with_pronunciation",
            {"raw": base_result, "previous_state": state.to_context(), "car_plate": base_plate},
        )

        suffix_result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=f"""
任务：复核用户最新音频中的车牌末尾是否包含特殊车牌尾字，并修正第一步更新结果。只输出 JSON 对象，字段为 car_plate。
当前识别状态：{json.dumps(state.to_context(), ensure_ascii=False)}。
第一步更新结果：{base_plate}

这一步只重点检查车牌最后一位。用户的车牌末尾可能不是数字或字母，而是特殊车牌尾字。

### 特殊车牌尾字

| 尾字 | 含义 |
|------|------|
| 警 | 警车 |
| 临 | 临时车牌 |
| 学 | 教练车 |
| 领 | 领事馆车牌 |
| 挂 | 半挂车牌 |

如果用户最新音频里明确说了这些特殊尾字，car_plate 末尾要保留对应汉字。
如果用户最新音频里没有明确特殊尾字，就保持第一步更新结果。

### 临字重点规则

“临”是临时车牌尾字，语音里可能被听成“零”“0”“洞”“林”“拎”“令”。
如果用户是在车牌末尾说“临”“临牌”“临时牌”“临时车牌”，或者末尾发音明显接近“临”，必须把 car_plate 最后一位改成“临”，不要输出 0。
如果用户明确是在报数字“零”或“洞”，并且不是临时车牌语义，才保留数字 0。

### 示例

第一步更新结果：粤B12340
用户最新音频末尾说的是：粤 B 一二三四 临
输出：{{"car_plate":"粤B1234临"}}

第一步更新结果：京A88880
用户最新音频末尾说的是：京 A 八八八八 临牌
输出：{{"car_plate":"京A8888临"}}

第一步更新结果：沪C12340
用户最新音频末尾说的是：沪 C 一二三四 洞
输出：{{"car_plate":"沪C12340"}}
""",
            max_tokens=128,
        )
        suffix_plate = clean_plate_text(parse_json_object(suffix_result).get("car_plate")) or base_plate
        car_plate = await self.normalize_plate_result(
            model=model,
            wav_bytes=wav_bytes,
            car_plate=suffix_plate,
            node="update_car_plate.normalize",
        )
        log_node_output(
            "update_car_plate.step3_special_suffix",
            {"raw": suffix_result, "previous_car_plate": base_plate, "step3_car_plate": suffix_plate, "car_plate": car_plate},
        )
        log_node_output(
            "update_car_plate",
            {
                "step1_car_plate": base_plate,
                "step3_car_plate": car_plate,
                "car_plate": car_plate,
                "previous_state": state.to_context(),
            },
        )
        return car_plate

    async def normalize_plate_result(
        self,
        *,
        model: str,
        wav_bytes: bytes,
        car_plate: str,
        node: str,
    ) -> str:
        formatted_plate = normalize_plate_format(car_plate)
        corrected_plate = formatted_plate
        retry_raw = ""
        retry_plate = ""
        if first_char_is_ascii_letter_or_digit(formatted_plate):
            retry_raw = await self.audio_call(
                model=model,
                wav_bytes=wav_bytes,
                prompt=f"""
任务：当前暂时的车牌识别结果第一位不是省份简称，请根据用户音频重新识别车牌号。
当前暂时识别结果：{formatted_plate}

请输出带省份简称的完整车牌号码。
车牌第一位必须是省份简称，例如：京、津、冀、晋、蒙、辽、吉、黑、沪、苏、浙、皖、闽、赣、鲁、豫、鄂、湘、粤、桂、琼、渝、川、贵、云、藏、陕、甘、青、宁、新。
只输出 JSON 对象，字段为 car_plate。
""",
                max_tokens=128,
            )
            retry_plate = normalize_plate_format(parse_json_object(retry_raw).get("car_plate"))
            if retry_plate:
                corrected_plate = retry_plate
        final_plate = replace_leading_g_with_ji(corrected_plate)
        log_node_output(
            node,
            {
                "input_car_plate": car_plate,
                "formatted_car_plate": formatted_plate,
                "province_retry_used": bool(retry_raw),
                "province_retry_raw": retry_raw,
                "province_retry_car_plate": retry_plate,
                "car_plate": final_plate,
            },
        )
        return final_plate

    async def generate_reply(self, *, model: str, state: PlateAgentState, changed: bool) -> str:
        result, status_code = await self.model_client.complete_text(
            model=model or QWEN_MODEL,
            text=json.dumps(
                {
                    "car_plate": state.car_plate,
                    "vehicle_type": state.vehicle_type,
                    "confusions": [item.to_dict() for item in with_relative_confusion_reasons(state.car_plate, state.confusions)],
                    "changed": changed,
                },
                ensure_ascii=False,
            ),
            prompt=(
                "任务：根据当前暂时识别的车牌、易混淆字符列表、车辆类型，生成口语化客服回复。"
                "回复要求：1. 说出当前识别到的车牌；"
                "2. 易混淆字符列表 confusions 是必须重点确认的内容，不能忽略、合并或省略；"
                "3. 如果 confusions 不为空，必须逐项按 reason 的描述向用户确认当前识别结果；"
                "4. 不要把 position 说给用户，不能说“第几位”“第1位”“第2位”等绝对位置；"
                "5. 如果同一类易混淆字符出现多次，要使用 reason 里的“前面的、后面的、靠前的、靠后的”等相对说法区分；"
                "6. 如果某一项 candidates 为空，不要编造候选值，不要说“是 A 还是 B”，只说当前识别为 reason 里的内容并请用户确认；"
                "7. 如果某一项 candidates 不为空，才可以带候选值请用户确认；"
                "8. 如果有多个易混淆字符，要全部说出来，不能只确认其中一个；"
                "9. 如果是 8 位新能源号牌，要询问用户是不是新能源电车；"
                "10. 简短自然，不要解释系统逻辑。"
                "只输出 JSON 对象，字段为 car_plate 和 assistant_reply。"
            ),
            history=[],
            max_tokens=256,
            output_audio=False,
        )
        if status_code >= 400:
            reply = fallback_reply(state)
            log_node_output(
                "generate_reply",
                {
                    "status_code": status_code,
                    "raw": result.get("text"),
                    "assistant_reply": reply,
                    "fallback_used": True,
                    "state": state.to_context(),
                },
            )
            return reply
        parsed = parse_json_object(result.get("text"))
        parsed_reply = str(parsed.get("assistant_reply") or "").strip()
        fallback_used = not parsed_reply or contains_absolute_position_text(parsed_reply)
        reply = parsed_reply if not fallback_used else fallback_reply(state)
        log_node_output(
            "generate_reply",
            {
                "status_code": status_code,
                "raw": result.get("text"),
                "assistant_reply": reply,
                "fallback_used": fallback_used,
                "state": state.to_context(),
            },
        )
        return reply

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
    confusions: list[PlateConfusion] = []
    for item in state.confusions:
        cloned = PlateConfusion.from_value(item.to_dict())
        if cloned is not None:
            confusions.append(cloned)
    return PlateAgentState(
        car_plate=state.car_plate,
        vehicle_type=state.vehicle_type,
        confusions=confusions,
        final_car_plate=state.final_car_plate,
        assistant_reply=state.assistant_reply,
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


def parse_json_object(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(extract_json_candidate(raw))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def parse_bool_text(text: str, *, default: bool) -> bool:
    value = str(text or "").strip().lower()
    if "true" in value:
        return True
    if "false" in value:
        return False
    return default


def parse_yes_no(text: str, *, default: bool) -> bool:
    value = str(text or "").strip().lower()
    if re.search(r"\byes\b", value) or "true" in value:
        return True
    if re.search(r"\bno\b", value) or "false" in value:
        return False
    return default


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


def detect_initial_confusions_by_rule(car_plate: str) -> list[PlateConfusion]:
    plate = clean_plate_text(car_plate)
    confusions: list[PlateConfusion] = []
    if plate:
        first = plate[0]
        if first in {"甘", "赣"}:
            confusions.append(
                PlateConfusion(
                    position=1,
                    value=first,
                    reason=f"第1位当前识别为{describe_plate_char(first)}，请用户确认。",
                )
            )
        if first in {"津", "京"}:
            confusions.append(
                PlateConfusion(
                    position=1,
                    value=first,
                    reason=f"第1位当前识别为{describe_plate_char(first)}，请用户确认。",
                )
            )
        if first in {"桂", "贵"}:
            confusions.append(
                PlateConfusion(
                    position=1,
                    value=first,
                    reason=f"第1位当前识别为{describe_plate_char(first)}，请用户确认。",
                )
            )
        if first in {"冀", "吉"}:
            confusions.append(
                PlateConfusion(
                    position=1,
                    value=first,
                    reason=f"第1位当前识别为{describe_plate_char(first)}，请用户确认。",
                )
            )
    for index, value in enumerate(plate, start=1):
        if value in {"2", "R"}:
            confusions.append(
                PlateConfusion(
                    position=index,
                    value=value,
                    reason=f"第{index}位当前识别为{describe_plate_char(value)}，请用户确认。",
                )
            )
        if value in {"1", "E"}:
            confusions.append(
                PlateConfusion(
                    position=index,
                    value=value,
                    reason=f"第{index}位当前识别为{describe_plate_char(value)}，请用户确认。",
                )
            )
    return with_relative_confusion_reasons(plate, confusions)


def with_relative_confusion_reasons(car_plate: str, confusions: list[PlateConfusion]) -> list[PlateConfusion]:
    plate = clean_plate_text(car_plate)
    normalized: list[PlateConfusion] = []
    for item in confusions:
        normalized.append(
            PlateConfusion(
                position=item.position,
                value=item.value,
                candidates=list(item.candidates),
                reason=relative_confusion_reason(plate, item),
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
        if index == 0:
            return f"车牌开头的省份简称当前识别为{value_label}，请用户确认。"
        duplicate_label = duplicate_confusion_label(plate, index, value)
        if duplicate_label:
            return f"车牌里{duplicate_label}，请用户确认。"
        if index == len(plate) - 1:
            return f"车牌末尾当前识别为{value_label}，请用户确认。"
        if index <= 2:
            return f"车牌靠前位置当前识别为{value_label}，请用户确认。"
        if index >= len(plate) - 3:
            return f"车牌靠后位置当前识别为{value_label}，请用户确认。"
        return f"车牌中间部分当前识别为{value_label}，请用户确认。"
    return f"车牌中当前识别为{value_label}的位置，请用户确认。"


def duplicate_confusion_label(car_plate: str, index: int, value: str) -> str:
    key = confusion_group_key(value)
    if not key:
        return ""
    positions = [idx for idx, char in enumerate(car_plate) if confusion_group_key(char) == key]
    if len(positions) <= 1 or index not in positions:
        return ""
    rank = positions.index(index)
    label = describe_plate_char(value)
    if len(positions) == 2:
        prefix = "前面的" if rank == 0 else "后面的"
    elif rank == 0:
        prefix = "最前面的"
    elif rank == len(positions) - 1:
        prefix = "最后面的"
    else:
        prefix = "中间的"
    return f"{prefix}{label}"


def confusion_group_key(value: str) -> str:
    value = str(value or "").upper()
    if value in {"2", "R"}:
        return "2/R"
    if value in {"1", "E"}:
        return "1/E"
    if value in {"甘", "赣"}:
        return "甘/赣"
    if value in {"津", "京"}:
        return "津/京"
    if value in {"桂", "贵"}:
        return "桂/贵"
    if value in {"冀", "吉"}:
        return "冀/吉"
    return ""


def describe_plate_char(value: str) -> str:
    labels = {
        "赣": "江西的赣",
        "甘": "甘肃的甘",
        "津": "天津的津",
        "京": "北京的京",
        "桂": "广西的桂",
        "贵": "贵州的贵",
        "冀": "河北的冀",
        "吉": "吉林的吉",
    }
    if value in labels:
        return labels[value]
    if value.isdigit():
        return f"数字 {value}"
    if re.fullmatch(r"[A-Za-z]", value):
        return f"字母 {value.upper()}"
    return value


def fallback_reply(state: PlateAgentState) -> str:
    plate = state.car_plate or "当前车牌"
    parts = [f"我这边暂时识别到的车牌号是{plate}。"]
    if state.confusions:
        descriptions = [item.reason.rstrip("。") for item in with_relative_confusion_reasons(state.car_plate, state.confusions)]
        parts.append("请您再确认一下：" + "；".join(descriptions) + "。")
    else:
        parts.append("请您确认一下是否正确。")
    if state.vehicle_type == "new_energy":
        parts.append("另外这是新能源号牌吗？")
    return "".join(parts)


def contains_absolute_position_text(value: str) -> bool:
    return bool(re.search(r"第\s*[0-9一二三四五六七八九十]+\s*位", str(value or "")))


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


plate_agent_service: PlateAgentService | None = None


def get_plate_agent_service(model_client: ChatModel) -> PlateAgentService:
    global plate_agent_service
    if plate_agent_service is None or plate_agent_service.model_client is not model_client:
        plate_agent_service = PlateAgentService(model_client)
    return plate_agent_service
