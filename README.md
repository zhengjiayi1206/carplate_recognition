# Carplate Chatbox

This project contains only the `/chatbox` realtime audio chat route. It calls a vLLM/OpenAI-compatible Qwen endpoint directly through `QWEN_API_BASE`.

## Routes

- `GET /` redirects to `/chatbox`
- `GET /chatbox`
- `GET /health`
- `POST /api/chatbox/session/start`
- `POST /api/chatbox/text`
- `POST /api/chatbox/audio`
- `POST /api/chatbox/speech`
- `WS /ws/audio`
- `WS /ws/vad`

FastAPI docs and other demo pages are disabled.

See `API_AUDIO.md` for the non-realtime audio API calling sequence.

## Run

Install dependencies first:

```bash
chmod +x setup_uv.sh
./setup_uv.sh
```

Start the chatbox service:

```bash
chmod +x run_chatbox.sh stop_chatbox.sh
./run_chatbox.sh
```

Stop it:

```bash
./stop_chatbox.sh
```

The system prompt is loaded from `realtime_audio_demo/system_prompt.md`.
The file can be mounted or overwritten on the server container.

Open:

```text
http://127.0.0.1:55785/chatbox
```

By default, voice input is cut by VAD, converted to WAV, and sent directly to the vLLM/OpenAI-compatible chat completions endpoint.

EasyTurn is disabled by default. The local EasyTurn model implementation is not shipped in this project.
To enable it, deploy EasyTurn as a separate HTTP service and configure:

```bash
export EASYTURN_ENABLED=1
export EASYTURN_API_URL=http://easyturn-server:8000/judge
./run_chatbox.sh
```

The EasyTurn API should accept `audio_base64`, `audio_format`, `source_sample_rate`, and `target_sample_rate`, then return `turn_state` such as `INCOMPLETE` or `COMPLETE`.

The service expects a Qwen-compatible server at `QWEN_API_BASE`, defaulting to:

```text
http://127.0.0.1:5440/v1
```

Default model name:

```text
qwen3_omni
```
