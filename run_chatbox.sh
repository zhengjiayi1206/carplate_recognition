#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export QWEN_API_BASE="${QWEN_API_BASE:-http://127.0.0.1:5440/v1}"
export QWEN_MODEL="${QWEN_MODEL:-qwen3-omni}"
export QWEN_PROVIDER="${QWEN_PROVIDER:-vllm_omni}"
export PREFILL_INTERVAL_MS="${PREFILL_INTERVAL_MS:-600}"
export PREFILL_MODE="${PREFILL_MODE:-cumulative_probe}"
export TARGET_SAMPLE_RATE="${TARGET_SAMPLE_RATE:-16000}"
export FINAL_MAX_TOKENS="${FINAL_MAX_TOKENS:-512}"
export MAX_HISTORY_TURNS="${MAX_HISTORY_TURNS:-10}"
export STREAM_FINAL_OUTPUT="${STREAM_FINAL_OUTPUT:-1}"
export TTS_API_BASE="${TTS_API_BASE:-http://127.0.0.1:5446}"
export TTS_MODEL="${TTS_MODEL:-cosyvoice3}"
export TTS_VOICE="${TTS_VOICE:-}"
export TTS_RESPONSE_FORMAT="${TTS_RESPONSE_FORMAT:-wav}"
export TTS_STREAM_RESPONSE_FORMAT="${TTS_STREAM_RESPONSE_FORMAT:-pcm}"
export TTS_STREAM_FORMAT="${TTS_STREAM_FORMAT:-audio}"
export TTS_SAMPLE_RATE="${TTS_SAMPLE_RATE:-24000}"
export TTS_TASK_TYPE="${TTS_TASK_TYPE:-}"
export TTS_REF_AUDIO="${TTS_REF_AUDIO:-}"
export TTS_REF_TEXT="${TTS_REF_TEXT:-}"
export EASYTURN_ENABLED="${EASYTURN_ENABLED:-0}"
export EASYTURN_API_URL="${EASYTURN_API_URL:-}"
export EASYTURN_ACK_TEXT="${EASYTURN_ACK_TEXT:-嗯，我在听，你继续。}"
export SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-${SCRIPT_DIR}/realtime_audio_demo/system_prompt.md}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-56010}"
PID_FILE="${PID_FILE:-${SCRIPT_DIR}/chatbox.pid}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/chatbox.log}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "chatbox already running: pid=${old_pid}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

if command -v uv >/dev/null 2>&1; then
  RUNNER=(uv run python -m uvicorn app:app)
else
  RUNNER=(python -m uvicorn app:app)
fi

nohup "${RUNNER[@]}" --host "${HOST}" --port "${PORT}" --log-level info > "${LOG_FILE}" 2>&1 &
pid="$!"
echo "${pid}" > "${PID_FILE}"

echo "chatbox started: pid=${pid}"
echo "url: http://127.0.0.1:${PORT}/chatbox"
echo "log: ${LOG_FILE}"
