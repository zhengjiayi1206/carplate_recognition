from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from realtime_audio_demo.config import QWEN_MODEL
from realtime_audio_demo.services.interfaces import ChatModel
from realtime_audio_demo.services.output_filter import extract_json_candidate


logger = logging.getLogger("uvicorn.error")

NO_PLATE_REPLY = "我没有听到车牌号内容，请告诉我车牌号。"
INVALID_PLATE_REPLY = "您好，您当前的车牌号并不是有效号码，请重新输入。"

CAR_PLATE_EXTRACTION_PROMPT = """
你是一个中文语音车牌号抽取器。

任务：从用户音频中识别并抽取车牌号。
最终只能输出一行严格 JSON，不要解释、不要 Markdown、不要多余文字。

输出格式必须是：
{"car_plate":"..."}

如果音频中没有明确车牌号，输出：
{"car_plate":""}

### 抽取范围
只抽取用户说出的车牌号本身，忽略以下无关表达：
我的车牌是车牌号是应该是帮我登记你听一下确认一下等。

### 中国常见车牌格式
车牌通常为 7 位或 8 位：
1. 第 1 位：省份简称
2. 第 2 位：英文字母，通常是发牌机关代码
3. 后 5 到 6 位：数字或大写英文字母

常见省份简称包括：
京、津、冀、晋、蒙、辽、吉、黑、沪、苏、浙、皖、闽、赣、鲁、豫、鄂、湘、粤、桂、琼、渝、川、贵、云、藏、陕、甘、青、宁、新。

### 识别规则
1. 只输出车牌号，不输出用户原话。
2. 英文字母统一输出大写。
3. 删除车牌中的空格、顿号、逗号、横杠等分隔符。
4. 中文数字要转换为阿拉伯数字，例如：一1，二2，三3，四4，五5，六6，七7，八8，九9，零/洞0。
5. 如果听到字母A / A / 诶，输出 A；字母B / B / 比，输出 B，以此类推。
6. 第 1 位必须优先识别为省份简称，不要把省份简称误写成同音字或英文字母。
7. 第 2 位通常是英文字母；如果第 2 位听起来像数字但整体更符合车牌格式，应优先判断为相近英文字母。
8. 后 5 到 6 位根据发音判断为数字或英文字母，保证整体最像一个合法车牌。
9. 常见混淆要结合车牌格式判断：
   - 零 / 洞 / 欧 / O在后段更可能是 0
   - 一 / 幺 / 衣 / I在后段更可能是 1
   - B / 8G / 吉 / 冀E / 一 / 衣S / 4Z / 2A / 诶D / 弟F / 艾夫需要结合位置和格式判断
10. 如果音频中出现多个候选车牌，只输出最完整、最符合车牌格式的一个。
11. 如果只听到部分车牌，或者长度明显不足，不要猜完整，输出空字符串。
12. 如果听到的内容不是车牌号，不要强行生成，输出空字符串。

### 示例
用户说：我的车牌是粤 B 八六三二九
输出：{"car_plate":"粤B86329"}

用户说：车牌号琼 A 七四五三
输出：{"car_plate":"琼A7453"}

用户说：京 A D 三二一六八
输出：{"car_plate":"京AD32168"}

用户没有说车牌号
输出：{"car_plate":""}
""".strip()


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
        confusions = await self.detect_confusions(
            model=model,
            wav_bytes=wav_bytes,
            car_plate=working.car_plate,
            state=working,
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
        format_result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=CAR_PLATE_EXTRACTION_PROMPT,
            max_tokens=128,
        )
        format_plate = clean_plate_text(parse_json_object(format_result).get("car_plate"))
        log_node_output("extract_car_plate.step1_format", {"raw": format_result, "car_plate": format_plate})

        pronunciation_result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=f"""
任务：复核用户音频中的车牌字符发音，并修正第一次识别结果。只输出 JSON 对象，字段为 car_plate。
第一次识别结果：{format_plate}

这一步重点检查是否有多音、口语、报号习惯导致的数字或字母误识别。
用户说的是车牌字符的读音，不是要把这些读音汉字写进结果。

### 常见读音纠正规则

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

请结合用户音频和第一次识别结果，重新输出修正后的 car_plate。
car_plate 只能包含省份简称、英文字母、数字，不能包含洞、幺、是、陆、拐、吸、勾、沟儿、圈。
""",
            max_tokens=128,
        )
        pronunciation_plate = clean_plate_text(parse_json_object(pronunciation_result).get("car_plate")) or format_plate
        log_node_output(
            "extract_car_plate.step2_pronunciation",
            {"raw": pronunciation_result, "previous_car_plate": format_plate, "car_plate": pronunciation_plate},
        )

        suffix_result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=f"""
任务：复核用户音频中的车牌末尾是否包含特殊车牌尾字，并修正第二次识别结果。只输出 JSON 对象，字段为 car_plate。
第二次识别结果：{pronunciation_plate}

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
如果音频里没有明确特殊尾字，就保持第二次识别结果。
""",
            max_tokens=128,
        )
        suffix_plate = clean_plate_text(parse_json_object(suffix_result).get("car_plate")) or pronunciation_plate
        final_plate = await self.normalize_plate_result(
            model=model,
            wav_bytes=wav_bytes,
            car_plate=suffix_plate,
            node="extract_car_plate.normalize",
        )
        log_node_output(
            "extract_car_plate.step3_special_suffix",
            {"raw": suffix_result, "previous_car_plate": pronunciation_plate, "car_plate": final_plate},
        )
        log_node_output(
            "extract_car_plate",
            {
                "step1_car_plate": format_plate,
                "step2_car_plate": pronunciation_plate,
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
    ) -> list[PlateConfusion]:
        result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=(
                "任务：结合用户最新音频、上一轮 confusions、当前识别车牌，判断当前还需要二次确认的易混淆字符。"
                f"当前识别车牌：{car_plate}。"
                f"当前状态：{json.dumps((state or PlateAgentState(car_plate=car_plate)).to_context(), ensure_ascii=False)}。"
                "当前状态里的 confusions 是上一轮已经询问过用户、等待用户确认的混淆列表。"
                "处理顺序必须如下："
                "第一步，先根据用户最新音频判断上一轮 confusions 有没有被确认。"
                "如果用户没有明确说某一项不对、不是、改成、应该是、这一位错了，就认为上一轮该项已经被用户确认，本轮不要再输出该项。"
                "如果用户明确说上一轮某一项不对或进行了修正，再结合当前识别车牌判断这一位是否仍然需要二次确认。"
                "第二步，再从第 1 位到最后 1 位逐字符扫描完整的当前识别车牌，查找新出现、尚未被用户确认过的混淆字符。"
                "只要发现新的混淆字符，每一个命中的字符位置都必须单独输出一条 confusions 记录。"
                "不能只返回第一个命中的位置，不能合并多个位置。"
                "需要检查的混淆字符规则如下："
                "1. 如果某一位是 2 或 R，必须输出这一位，candidates 必须包含 2 和 R，reason 说明数字 2 和字母 R 容易混淆，需要用户确认；"
                "2. 如果某一位是 1 或 E，必须输出这一位，candidates 必须包含 1 和 E，reason 说明数字 1、幺 和字母 E 容易混淆，需要用户确认；"
                "3. 如果第一位是 甘 或 赣，必须输出第一位，reason 说明当前识别为甘肃的甘或江西的赣，需要用户确认；"
                "4. 如果第一位是 津 或 京，必须输出第一位，candidates 必须包含 津 和 京，reason 说明天津简称津和北京简称京发音接近，需要用户确认；"
                "5. 如果第一位是 桂 或 贵，必须输出第一位，reason 说明当前识别为广西的桂或贵州的贵，需要用户确认；"
                "6. 其他语音中明显不确定、用户发音含糊、背景噪声导致可能听错的位置，也要额外标记出来。"
                "只输出 JSON 对象，字段为 confusions。confusions 是数组，每一项包含 position、value、candidates、reason。"
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
        confusions = [item for item in items if item is not None]
        log_node_output(
            "detect_confusions",
            {
                "raw": result,
                "car_plate": car_plate,
                "confusions": [item.to_dict() for item in confusions],
            },
        )
        return confusions

    async def detect_confirmation(self, *, model: str, wav_bytes: bytes, state: PlateAgentState) -> bool:
        result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=(
                "任务：判断用户这次语音是在确认当前车牌正确，还是要求继续修改。"
                f"当前识别状态：{json.dumps(state.to_context(), ensure_ascii=False)}。"
                "如果用户表达对、是的、没错、正确、就是这个、确认等含义，回答 yes。"
                "如果用户表达不对、修改、不是、某一位错了、重新说车牌或补充内容，回答 no。"
                "只回答 yes 或 no，不要输出其它内容。"
            ),
            max_tokens=8,
        )
        confirmed = parse_yes_no(result, default=False)
        log_node_output("detect_confirmation", {"raw": result, "confirmed": confirmed, "state": state.to_context()})
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
            prompt=(
                "你正在更新已经识别过的车牌号，不是开始一个全新的会话。\n"
                f"上一轮识别状态：{update_context}\n\n"
                "任务：结合上一轮识别状态和用户最新音频，判断用户这次要更新什么。\n"
                "用户可能只修改某一位、确认某个混淆位、补充缺失字符，也可能重新报完整车牌。\n"
                "如果用户只修改某一位，必须在上一轮 car_plate 基础上只替换对应位置，并输出完整的新 car_plate。\n"
                "如果用户重新报了完整车牌，以用户最新音频为准输出完整的新 car_plate。\n"
                "如果用户没有提供可用于更新车牌的新内容，输出上一轮 car_plate。\n\n"
                + CAR_PLATE_EXTRACTION_PROMPT
            ),
            max_tokens=128,
        )
        base_plate = clean_plate_text(parse_json_object(base_result).get("car_plate")) or state.car_plate
        log_node_output(
            "update_car_plate.step1_update",
            {"raw": base_result, "previous_state": state.to_context(), "car_plate": base_plate},
        )

        pronunciation_result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=f"""
任务：复核用户最新音频中的车牌字符发音，并修正第一步更新结果。只输出 JSON 对象，字段为 car_plate。
当前识别状态：{json.dumps(state.to_context(), ensure_ascii=False)}。
第一步更新结果：{base_plate}

这一步重点检查用户这次修改中是否有多音、口语、报号习惯导致的数字或字母误识别。
用户说的是车牌字符的读音，不是要把这些读音汉字写进结果。

### 常见读音纠正规则

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

请结合用户最新音频、当前识别状态和第一步更新结果，重新输出修正后的 car_plate。
car_plate 只能包含省份简称、英文字母、数字，不能包含洞、幺、是、陆、拐、吸、勾、沟儿、圈。
""",
            max_tokens=128,
        )
        pronunciation_plate = clean_plate_text(parse_json_object(pronunciation_result).get("car_plate")) or base_plate
        log_node_output(
            "update_car_plate.step2_pronunciation",
            {"raw": pronunciation_result, "previous_car_plate": base_plate, "car_plate": pronunciation_plate},
        )

        suffix_result = await self.audio_call(
            model=model,
            wav_bytes=wav_bytes,
            prompt=f"""
任务：复核用户最新音频中的车牌末尾是否包含特殊车牌尾字，并修正第二步更新结果。只输出 JSON 对象，字段为 car_plate。
当前识别状态：{json.dumps(state.to_context(), ensure_ascii=False)}。
第二步更新结果：{pronunciation_plate}

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
如果用户最新音频里没有明确特殊尾字，就保持第二步更新结果。
""",
            max_tokens=128,
        )
        suffix_plate = clean_plate_text(parse_json_object(suffix_result).get("car_plate")) or pronunciation_plate
        car_plate = await self.normalize_plate_result(
            model=model,
            wav_bytes=wav_bytes,
            car_plate=suffix_plate,
            node="update_car_plate.normalize",
        )
        log_node_output(
            "update_car_plate.step3_special_suffix",
            {"raw": suffix_result, "previous_car_plate": pronunciation_plate, "step3_car_plate": suffix_plate, "car_plate": car_plate},
        )
        log_node_output(
            "update_car_plate",
            {
                "step1_car_plate": base_plate,
                "step2_car_plate": pronunciation_plate,
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
                    "confusions": [item.to_dict() for item in state.confusions],
                    "changed": changed,
                },
                ensure_ascii=False,
            ),
            prompt=(
                "任务：根据当前暂时识别的车牌、易混淆字符列表、车辆类型，生成口语化客服回复。"
                "回复要求：1. 说出当前识别到的车牌；"
                "2. 易混淆字符列表 confusions 是必须重点确认的内容，不能忽略、合并或省略；"
                "3. 如果 confusions 不为空，必须逐项按 reason 的描述向用户确认当前识别结果；"
                "4. 如果某一项 candidates 为空，不要编造候选值，不要说“是 A 还是 B”，只说当前识别为 reason 里的内容并请用户确认；"
                "5. 如果某一项 candidates 不为空，才可以带候选值请用户确认；"
                "6. 如果有多个易混淆字符，要全部说出来，不能只确认其中一个；"
                "7. 如果是 8 位新能源号牌，要询问用户是不是新能源电车；"
                "8. 简短自然，不要解释系统逻辑。"
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
        fallback_used = not parsed_reply
        reply = parsed_reply or fallback_reply(state)
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
    return confusions


def describe_plate_char(value: str) -> str:
    labels = {
        "赣": "江西的赣",
        "甘": "甘肃的甘",
        "津": "天津的津",
        "京": "北京的京",
        "桂": "广西的桂",
        "贵": "贵州的贵",
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
        descriptions = []
        for item in state.confusions:
            label = f"第{item.position}位" if item.position else "其中一位"
            descriptions.append(f"{label}{item.value}")
        parts.append("请您再确认一下" + "、".join(descriptions) + "。")
    else:
        parts.append("请您确认一下是否正确。")
    if state.vehicle_type == "new_energy":
        parts.append("另外这是新能源号牌吗？")
    return "".join(parts)


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


plate_agent_service: PlateAgentService | None = None


def get_plate_agent_service(model_client: ChatModel) -> PlateAgentService:
    global plate_agent_service
    if plate_agent_service is None or plate_agent_service.model_client is not model_client:
        plate_agent_service = PlateAgentService(model_client)
    return plate_agent_service
