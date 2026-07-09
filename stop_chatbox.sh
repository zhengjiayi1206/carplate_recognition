#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${PID_FILE:-${SCRIPT_DIR}/chatbox.pid}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "chatbox is not running: pid file not found"
  exit 0
fi

pid="$(cat "${PID_FILE}" || true)"
if [[ -z "${pid}" ]]; then
  rm -f "${PID_FILE}"
  echo "chatbox is not running: empty pid file removed"
  exit 0
fi

if ! kill -0 "${pid}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "chatbox is not running: stale pid file removed"
  exit 0
fi

kill "${pid}"
for _ in $(seq 1 20); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "chatbox stopped: pid=${pid}"
    exit 0
  fi
  sleep 0.5
done

kill -TERM "${pid}" 2>/dev/null || true
sleep 1
if kill -0 "${pid}" 2>/dev/null; then
  kill -KILL "${pid}" 2>/dev/null || true
fi
rm -f "${PID_FILE}"
echo "chatbox stopped: pid=${pid}"
