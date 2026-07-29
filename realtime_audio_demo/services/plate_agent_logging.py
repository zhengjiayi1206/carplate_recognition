from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime
from contextvars import ContextVar
from typing import Any

from realtime_audio_demo.config import (
    PLATE_AGENT_LOG_DETAIL_MAX_CHARS,
    PLATE_AGENT_TRACE_DIR,
    PLATE_AGENT_TRACE_ENABLED,
)


logger = logging.getLogger("uvicorn.error")
CURRENT_SESSION_ID: ContextVar[str] = ContextVar("plate_agent_session_id", default="")
CURRENT_TURN_BEFORE_STATE: ContextVar[dict[str, Any] | None] = ContextVar("plate_agent_turn_before_state", default=None)
_TRACE_LOCK = threading.Lock()
_TRACE_FILENAMES_BY_SESSION: dict[str, str] = {}


def state_change_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keys = [
        "car_plate",
        "confirmed",
        "final_car_plate",
        "vehicle_type",
        "plate_chars",
        "need_confirm_chars",
        "confirmed_chars",
    ]
    changes: dict[str, dict[str, Any]] = {}
    for key in keys:
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value != after_value:
            changes[key] = {
                "before": before_value,
                "after": after_value,
            }
    return changes


def log_node_output(node: str, output: dict[str, Any]) -> None:
    """普通服务日志：记录节点摘要，不写详细 JSONL 轨迹。"""

    before_state = output.get("before_state") or CURRENT_TURN_BEFORE_STATE.get()
    after_state = output.get("after_state") or output.get("state")
    payload: dict[str, Any] = {
        "session_id": CURRENT_SESSION_ID.get() or None,
        "method": node,
        "event": "node_output",
        "output": output,
    }
    if isinstance(before_state, dict):
        payload["before_state"] = before_state
    if isinstance(after_state, dict):
        payload["after_state"] = after_state
    if isinstance(before_state, dict) and isinstance(after_state, dict):
        payload["state_diff"] = state_change_summary(before_state, after_state)
    logger.info("plate_agent %s", format_node_log_summary(payload))


def log_agent_line(message: str, **fields: Any) -> None:
    """普通服务日志：记录中文摘要，不写详细 JSONL 轨迹。"""

    payload: dict[str, Any] = {
        "session_id": CURRENT_SESSION_ID.get() or None,
        "event": "trace_line",
        "说明": message,
    }
    if fields:
        payload["详情"] = fields
    logger.info("plate_agent_trace %s", format_trace_line_summary(payload))


def log_session_event(event: str, **fields: Any) -> None:
    """详细 session JSON：一个文件保存一个会话的真实 Agent 轨迹。"""

    session_id = CURRENT_SESSION_ID.get() or None
    history_entries = build_history_entries(event, fields)
    for history_entry in history_entries:
        write_session_trace(session_id=session_id, history_entry=history_entry)
    history_summary = history_entries[-1] if history_entries else {}
    payload: dict[str, Any] = {
        "session_id": session_id,
        "history": history_summary,
    }
    logger.info("plate_agent_event %s", format_session_event_summary(payload))


def build_history_entries(event: str, fields: dict[str, Any]) -> list[dict[str, Any]]:
    if event == "llm_request":
        status_bar = str(fields.get("status_bar") or "").strip()
        turn_instruction = str(fields.get("turn_instruction") or "").strip()
        entries: list[dict[str, Any]] = []
        if fields.get("input_type") == "audio":
            content: list[dict[str, Any]] = []
            request_text = status_bar or turn_instruction
            if request_text:
                content.append(
                    {
                        "type": "text",
                        "text": request_text,
                    }
                )
            content.append(
                {
                    "type": "audio",
                    "audio_bytes": fields.get("audio_bytes") or 0,
                }
            )
            entries.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        elif turn_instruction:
            entries.append({"role": "user", "content": turn_instruction})
        return entries
    return [build_history_entry(event, fields)]


def build_history_entry(event: str, fields: dict[str, Any]) -> dict[str, Any]:
    if event == "turn_start":
        return {
            "role": "user",
            "content": {
                "type": "audio_turn",
                "stage": fields.get("stage"),
                "audio_bytes": fields.get("audio_bytes"),
                "state": fields.get("state"),
                "turn_summaries": fields.get("turn_summaries") or [],
            },
        }
    if event == "llm_response":
        return {
            "role": "assistant",
            "content": fields.get("raw_output") or "",
        }
    if event == "tool_call":
        return {
            "role": "tool",
            "name": fields.get("tool"),
            "tool_call_id": fields.get("tool_call_id"),
            "content": {
                "arguments": fields.get("arguments") or {},
                "success": fields.get("success"),
                "message": fields.get("message"),
                "result": fields.get("tool_result"),
            },
        }
    if event == "final_response":
        return {
            "role": "assistant",
            "content": fields.get("speech_text") or "",
        }
    return {
        "role": "system",
        "content": {
            "type": event,
            **fields,
        },
    }


def write_session_trace(*, session_id: str | None, history_entry: dict[str, Any]) -> None:
    if not PLATE_AGENT_TRACE_ENABLED:
        return
    normalized_session_id = str(session_id or "no-session").strip() or "no-session"
    try:
        with _TRACE_LOCK:
            trace_path = PLATE_AGENT_TRACE_DIR / trace_filename_for_session(normalized_session_id)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            document = read_trace_document(trace_path, normalized_session_id)
            history = document.get("history")
            if not isinstance(history, list):
                history = []
                document["history"] = history
            history.append(history_entry)
            trace_path.write_text(json.dumps(document, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("plate_agent trace write failed session_id=%s error=%s", normalized_session_id, exc)


def read_trace_document(trace_path: Any, session_id: str) -> dict[str, Any]:
    if not trace_path.exists():
        return {"session_id": session_id, "history": []}
    try:
        value = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"session_id": session_id, "history": []}
    if not isinstance(value, dict):
        return {"session_id": session_id, "history": []}
    value["session_id"] = value.get("session_id") or session_id
    if not isinstance(value.get("history"), list):
        value["history"] = []
    return value


def safe_session_filename(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id)[:120] or "no-session"


def trace_filename_for_session(session_id: str) -> str:
    """同一个 session 固定写入同一个按时间命名的 JSON 文件。"""
    safe_session_id = safe_session_filename(session_id)
    filename = _TRACE_FILENAMES_BY_SESSION.get(safe_session_id)
    if filename:
        return filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{timestamp}_{safe_session_id}.json"
    _TRACE_FILENAMES_BY_SESSION[safe_session_id] = filename
    return filename


def format_node_log_summary(payload: dict[str, Any]) -> str:
    output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
    before_state = payload.get("before_state") if isinstance(payload.get("before_state"), dict) else {}
    after_state = payload.get("after_state") if isinstance(payload.get("after_state"), dict) else {}
    state = after_state or (output.get("state") if isinstance(output.get("state"), dict) else {})
    fields = {
        "session_id": payload.get("session_id"),
        "method": payload.get("method"),
        "stage": output.get("stage"),
        "action": output.get("action"),
        "car_plate": state.get("car_plate") or output.get("car_plate"),
        "before_plate": before_state.get("car_plate"),
        "after_plate": after_state.get("car_plate"),
        "status": output.get("task_status") or output.get("status"),
        "latency_ms": output.get("latency_ms"),
        "pending_count": len(state.get("need_confirm_chars") or []),
        "confirmed_count": len(state.get("confirmed_chars") or []),
    }
    return compact_kv(fields)


def format_trace_line_summary(payload: dict[str, Any]) -> str:
    details = payload.get("详情") if isinstance(payload.get("详情"), dict) else {}
    fields = {
        "session_id": payload.get("session_id"),
        "message": payload.get("说明"),
        "car_plate": first_present(details, "当前车牌", "候选车牌", "执行后车牌"),
        "stage": details.get("阶段") or details.get("stage"),
        "action": details.get("action"),
        "detail": short_text(select_detail_text(details)),
    }
    return compact_kv(fields)


def format_session_event_summary(payload: dict[str, Any]) -> str:
    history = payload.get("history") if isinstance(payload.get("history"), dict) else {}
    content = history.get("content") if isinstance(history.get("content"), dict) else {}
    fields = {
        "session_id": payload.get("session_id"),
        "role": history.get("role"),
        "name": history.get("name"),
        "success": content.get("success"),
    }
    return compact_kv(fields)


def compact_kv(fields: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        if value is None or value == "" or value == []:
            continue
        parts.append(f"{key}={short_text(value)}")
    return " ".join(parts)


def first_present(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if not is_empty_value(value):
            return value
    return None


def is_empty_value(value: Any) -> bool:
    return value is None or value == "" or value == []


def select_detail_text(details: dict[str, Any]) -> Any:
    for key in ("说明", "原因", "错误信息", "回复", "模型判断", "是否修改", "是否还有未处理修改", "编辑是否有效"):
        if key in details:
            return details.get(key)
    return ""


def short_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, default=str)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    max_chars = max(80, PLATE_AGENT_LOG_DETAIL_MAX_CHARS)
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text
