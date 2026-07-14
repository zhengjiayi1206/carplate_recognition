import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = APP_DIR / "static"
CAPTURE_DIR = APP_DIR / "captures"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
SYSTEM_PROMPT_PATH = Path(os.getenv("SYSTEM_PROMPT_PATH", str(APP_DIR / "realtime_audio_demo" / "system_prompt.md")))

QWEN_API_BASE = os.getenv("QWEN_API_BASE", "http://127.0.0.1:5440/v1").rstrip("/")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3_omni")
QWEN_PROVIDER = os.getenv("QWEN_PROVIDER", "auto")
QWEN_MODALITIES = [
    item.strip() for item in os.getenv("QWEN_MODALITIES", "text,audio").split(",") if item.strip()
]
QWEN_SPEAKER = os.getenv("QWEN_SPEAKER", "")
TARGET_SAMPLE_RATE = int(os.getenv("TARGET_SAMPLE_RATE", "16000"))
DEFAULT_PREFILL_MS = int(os.getenv("PREFILL_INTERVAL_MS", "600"))
PREFILL_MODE = os.getenv("PREFILL_MODE", "cumulative_probe")
FINAL_MAX_TOKENS = int(os.getenv("FINAL_MAX_TOKENS", "512"))
REQUEST_TIMEOUT = float(os.getenv("QWEN_REQUEST_TIMEOUT", "180"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "2"))
STREAM_FINAL_OUTPUT = os.getenv("STREAM_FINAL_OUTPUT", "1").lower() not in {"0", "false", "off", "no"}
SILERO_VAD_ENABLED = os.getenv("SILERO_VAD_ENABLED", "1").lower() not in {"0", "false", "off", "no"}
SILERO_VAD_PRELOAD = os.getenv("SILERO_VAD_PRELOAD", "1").lower() not in {"0", "false", "off", "no"}
SILERO_VAD_ONNX = os.getenv("SILERO_VAD_ONNX", "0").lower() in {"1", "true", "on", "yes"}
SILERO_VAD_THRESHOLD = float(os.getenv("SILERO_VAD_THRESHOLD", "0.5"))
SILERO_VAD_MIN_SPEECH_MS = int(os.getenv("SILERO_VAD_MIN_SPEECH_MS", "180"))
SILERO_VAD_MIN_SILENCE_MS = int(os.getenv("SILERO_VAD_MIN_SILENCE_MS", "450"))
SILERO_VAD_MAX_SPEECH_MS = int(os.getenv("SILERO_VAD_MAX_SPEECH_MS", "30000"))
SILERO_VAD_SPEECH_PAD_MS = int(os.getenv("SILERO_VAD_SPEECH_PAD_MS", "30"))
TTS_API_BASE = os.getenv("TTS_API_BASE", "").rstrip("/")
TTS_MODEL = os.getenv("TTS_MODEL", "cosyvoice3")
TTS_VOICE = os.getenv("TTS_VOICE", "default")
TTS_RESPONSE_FORMAT = os.getenv("TTS_RESPONSE_FORMAT", "wav")
EASYTURN_ENABLED = os.getenv("EASYTURN_ENABLED", "0").lower() in {"1", "true", "on", "yes"}
EASYTURN_API_URL = os.getenv("EASYTURN_API_URL", "").rstrip("/")
EASYTURN_ACK_TEXT = os.getenv("EASYTURN_ACK_TEXT", "嗯，我在听，你继续。")
SESSION_TTL = int(os.getenv("SESSION_TTL", "1800"))


def resolved_provider() -> str:
    if QWEN_PROVIDER != "auto":
        return QWEN_PROVIDER
    if ":5440" in QWEN_API_BASE or ":8091" in QWEN_API_BASE or "vllm-omni" in QWEN_API_BASE:
        return "vllm_omni"
    return "ms_swift"


def normalize_model_name(model: str) -> str:
    if resolved_provider() == "vllm_omni" and "/" not in model:
        return QWEN_MODEL
    return model
