# Chatbox Audio API

This document describes the non-realtime audio API for clients that upload one complete audio turn at a time.

Base URL examples:

```text
http://server-host:55785
```

## Minimal curl Call Flow

### 1. Start Session

先创建一个 `session_id`，然后调用 `/api/chatbox/session/start`。

```bash
curl -X POST "http://127.0.0.1:55785/api/chatbox/session/start" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-123-carplate-001",
    "model": "qwen3_omni",
    "outputAudio": false
  }'
```

返回里的 `text` 是固定开场白：

```json
{
  "session_id": "user-123-carplate-001",
  "text": "您好，请告诉我您的车牌号。"
}
```

前端展示这个开场白后，开始录用户的第一段完整语音。

### 2. Send Audio File

`/api/chatbox/audio` 当前不是 `multipart/form-data` 文件上传接口。调用时需要先把本地 WAV 文件转成 base64，再放到 JSON 的 `audio_base64` 字段里。

Linux/macOS:

```bash
AUDIO_BASE64="$(base64 -w 0 ./user_turn.wav)"

curl -X POST "http://127.0.0.1:55785/api/chatbox/audio" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"user-123-carplate-001\",
    \"model\": \"qwen3_omni\",
    \"audio_base64\": \"${AUDIO_BASE64}\",
    \"outputAudio\": false
  }"
```

如果服务器上的 `base64` 不支持 `-w 0`，用这个命令：

```bash
AUDIO_BASE64="$(base64 ./user_turn.wav | tr -d '\n')"
```

Windows PowerShell:

```powershell
$audioBase64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\path\to\user_turn.wav"))
$body = @{
  session_id = "user-123-carplate-001"
  model = "qwen3_omni"
  audio_base64 = $audioBase64
  outputAudio = $false
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:55785/api/chatbox/audio" `
  -ContentType "application/json" `
  -Body $body
```

### 3. Continue Or End

同一轮业务一直复用同一个 `session_id`：

```text
start session
  -> record user_turn_1.wav
  -> call /api/chatbox/audio
  -> record user_turn_2.wav
  -> call /api/chatbox/audio with same session_id
  -> repeat until final_car_plate is returned
```

如果返回的 `text` 里面包含：

```json
{
  "car_plate": "琼ADE540",
  "assistant_reply": "好的，已确认您的车牌号是琼ADE540。",
  "final_car_plate": "琼ADE540"
}
```

说明车牌流程已经完成，前端可以结束本次会话。下一次新业务要换新的 `session_id`，重新调用 `/api/chatbox/session/start`。

## Overview

Use these endpoints for a normal non-realtime audio flow:

```text
1. POST /api/chatbox/session/start
2. POST /api/chatbox/audio
3. Repeat /api/chatbox/audio with the same session_id until the business flow ends
```

`/api/chatbox/audio` is not a streaming interface. The client records one full user utterance, converts it to WAV base64, sends it once, receives one response, then records the next turn if needed.

For realtime microphone chunk streaming, use `WS /ws/audio` instead.

## Session ID

The client should create and keep a stable `session_id`.

Rules:

- Same conversation: reuse the same `session_id`.
- New conversation: create a new `session_id`.
- If `session_id` changes, backend history changes.
- Current backend history is in memory. Service restart clears history.

Example:

```text
user-123-carplate-20260707-001
```

## 1. Start Session

Endpoint:

```http
POST /api/chatbox/session/start
Content-Type: application/json
```

Purpose:

- Starts a conversation.
- Returns fixed assistant opening text.
- Writes the opening assistant text into backend history.
- Starts a background prefix warmup request to the model.

The fixed opening text is:

```text
您好，请告诉我您的车牌号。
```

Request:

```json
{
  "session_id": "user-123-carplate-20260707-001",
  "model": "qwen3_omni",
  "outputAudio": false
}
```

Fields:

| Field | Required | Description |
| --- | --- | --- |
| `session_id` | Recommended | Conversation ID maintained by the client. |
| `model` | No | Defaults to backend `QWEN_MODEL`, currently `qwen3_omni`. |
| `outputAudio` | No | If `true`, backend returns TTS audio for the opening text. |

Response:

```json
{
  "session_id": "user-123-carplate-20260707-001",
  "text": "您好，请告诉我您的车牌号。",
  "audio_data_url": null,
  "history_text": "您好，请告诉我您的车牌号。",
  "speech_text": "您好，请告诉我您的车牌号。",
  "output_is_json": false
}
```

Client action:

- Show `text`.
- If `audio_data_url` is not null, play it.
- Then record the user's answer.

## 2. Send One Audio Turn

Endpoint:

```http
POST /api/chatbox/audio
Content-Type: application/json
```

### curl Example With an Audio File

The API receives JSON, so the client should read the WAV file and put its base64 content into `audio_base64`.

Linux/macOS:

```bash
AUDIO_BASE64="$(base64 -w 0 ./user_turn.wav)"

curl -X POST "http://127.0.0.1:55785/api/chatbox/audio" \
  -H "Content-Type: application/json" \
  -d "{
    \"session_id\": \"user-123-carplate-001\",
    \"model\": \"qwen3_omni\",
    \"audio_base64\": \"${AUDIO_BASE64}\",
    \"outputAudio\": false
  }"
```

If your `base64` command does not support `-w 0`, use:

```bash
AUDIO_BASE64="$(base64 ./user_turn.wav | tr -d '\n')"
```

Windows PowerShell:

```powershell
$audioBase64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\path\to\user_turn.wav"))
$body = @{
  session_id = "user-123-carplate-001"
  model = "qwen3_omni"
  audio_base64 = $audioBase64
  outputAudio = $false
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:55785/api/chatbox/audio" `
  -ContentType "application/json" `
  -Body $body
```

If the audio is already formatted as a data URL, `audio_base64` can also be:

```json
{
  "audio_base64": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAE..."
}
```

Purpose:

- Sends one complete user audio utterance.
- Backend runs the plate-recognition agent pipeline with the audio.
- Backend stores the user WAV audio as user history.
- Backend stores the assistant JSON response as assistant history.
- If the user confirms the plate, returned `text` contains `final_car_plate`.

Audio format:

- WAV bytes, base64 encoded.
- `audio_base64` can be plain base64 or a data URL like `data:audio/wav;base64,...`.
- This endpoint does not accept `multipart/form-data` file upload. Convert the file to base64 first.

Request:

```json
{
  "session_id": "user-123-carplate-20260707-001",
  "model": "qwen3_omni",
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAE...",
  "outputAudio": false
}
```

Fields:

| Field | Required | Description |
| --- | --- | --- |
| `audio_base64` | Yes | Complete WAV audio as base64 or data URL. |
| `session_id` | Recommended | Same ID used in `/api/chatbox/session/start`. |
| `model` | No | Defaults to backend `QWEN_MODEL`. |
| `outputAudio` | No | If `true`, backend returns assistant TTS audio. |
| `history` | No | Optional manual history if no `session_id` is used. Usually do not use this. |

Full request field shape:

```json
{
  "session_id": "user-123-carplate-001",
  "model": "qwen3_omni",
  "audio_base64": "WAV audio base64",
  "outputAudio": false,
  "history": []
}
```

Response example when no plate content is heard:

```json
{
  "text": "{\n  \"car_plate\": \"\",\n  \"assistant_reply\": \"我没有听到车牌号内容，请告诉我车牌号。\"\n}",
  "audio_data_url": null,
  "history_text": "{\n  \"car_plate\": \"\",\n  \"assistant_reply\": \"我没有听到车牌号内容，请告诉我车牌号。\"\n}",
  "speech_text": "我没有听到车牌号内容，请告诉我车牌号。",
  "user_history_text": "",
  "user_display_text": "",
  "output_is_json": true,
  "latency_ms": 1200,
  "ttft_ms": null
}
```

Response example when plate is confirmed:

```json
{
  "text": "{\n  \"car_plate\": \"琼ADE540\",\n  \"assistant_reply\": \"好的，已确认您的车牌号是琼ADE540。\",\n  \"final_car_plate\": \"琼ADE540\"\n}",
  "audio_data_url": null,
  "history_text": "{\n  \"car_plate\": \"琼ADE540\",\n  \"assistant_reply\": \"好的，已确认您的车牌号是琼ADE540。\",\n  \"final_car_plate\": \"琼ADE540\"\n}",
  "speech_text": "好的，已确认您的车牌号是琼ADE540。",
  "user_history_text": "",
  "user_display_text": "",
  "output_is_json": true,
  "latency_ms": 1300,
  "ttft_ms": null
}
```

Client action:

- Show `user_display_text` as the user's message if not empty.
- Show `text` as the assistant response.
- If `audio_data_url` is not null, play it.
- If `text` contains `final_car_plate`, the business flow can end.
- Otherwise, record the next user utterance and call `/api/chatbox/audio` again with the same `session_id`.

## 3. End Conversation

There is no required end API for the non-realtime audio flow.

The client ends the flow by stopping calls with that `session_id`.

If a new business flow starts, create a new `session_id` and call:

```http
POST /api/chatbox/session/start
```

again.

## Calling Order Example

```text
Client creates session_id
  ↓
POST /api/chatbox/session/start
  ↓
Show/play "您好，请告诉我您的车牌号。"
  ↓
Record full user utterance as WAV
  ↓
POST /api/chatbox/audio with same session_id
  ↓
Show assistant JSON
  ↓
If not confirmed: record next utterance and repeat /api/chatbox/audio
  ↓
If confirmed: read final_car_plate and finish
```

## JavaScript Example

```js
const baseUrl = "http://server-host:55785";
const sessionId = crypto.randomUUID();

async function startSession() {
  const response = await fetch(`${baseUrl}/api/chatbox/session/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      model: "qwen3_omni",
      outputAudio: false,
    }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function sendAudioTurn(wavBase64) {
  const response = await fetch(`${baseUrl}/api/chatbox/audio`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      model: "qwen3_omni",
      audio_base64: wavBase64,
      outputAudio: false,
    }),
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
```

## Notes

- `/api/chatbox/audio` expects complete WAV audio per request.
- The backend does not do VAD in `/api/chatbox/audio`; the client decides when one utterance starts and ends.
- If the client needs server-side VAD and realtime chunk streaming, use `WS /ws/audio`.
- Backend history is maintained by `session_id`.
- Backend history is currently memory-based; restart clears it.
- Backend keeps the latest `MAX_HISTORY_TURNS` turns. The default is `2`.
