import copy
import threading
from dataclasses import dataclass
from typing import Any

from realtime_audio_demo.config import (
    SILERO_VAD_MAX_SPEECH_MS,
    SILERO_VAD_MIN_SILENCE_MS,
    SILERO_VAD_MIN_SPEECH_MS,
    SILERO_VAD_ONNX,
    SILERO_VAD_SPEECH_PAD_MS,
    SILERO_VAD_THRESHOLD,
)
from realtime_audio_demo.utils.audio import float32_bytes_to_list, resample_linear


SILERO_SAMPLE_RATE = 16000
SILERO_WINDOW_SAMPLES = 512
SILERO_DEVICE = "cpu"


class SileroVadUnavailable(RuntimeError):
    pass


class SileroVadModelFactory:
    def __init__(self, *, use_onnx: bool) -> None:
        self.use_onnx = use_onnx
        self._torch: Any
        self._base_model: Any
        self._load_model()
        self._warm_up()

    @property
    def torch(self) -> Any:
        return self._torch

    def new_model(self) -> Any:
        try:
            model = copy.deepcopy(self._base_model)
        except Exception:
            model = self._load_fresh_model()
        if hasattr(model, "reset_states"):
            model.reset_states()
        return model

    def _load_model(self) -> None:
        self._torch = self._import_torch()
        self._base_model = self._load_fresh_model()
        if hasattr(self._base_model, "reset_states"):
            self._base_model.reset_states()

    def _load_fresh_model(self) -> Any:
        try:
            from silero_vad import load_silero_vad
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps
            raise SileroVadUnavailable(
                "silero-vad is not installed. Run `uv sync` or `pip install silero-vad` on the server."
            ) from exc

        try:
            model = load_silero_vad(onnx=self.use_onnx)
        except TypeError:
            model = load_silero_vad()
        if hasattr(model, "to"):
            model.to(SILERO_DEVICE)
        return model

    def _warm_up(self) -> None:
        if self.use_onnx:
            return
        model = self.new_model()
        frame = self._torch.zeros(SILERO_WINDOW_SAMPLES, dtype=self._torch.float32, device=SILERO_DEVICE)
        with self._torch.no_grad():
            model(frame, SILERO_SAMPLE_RATE)
        if hasattr(model, "reset_states"):
            model.reset_states()

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps
            raise SileroVadUnavailable(
                "torch is required by silero-vad. Run `uv sync` or `pip install silero-vad` on the server."
            ) from exc
        return torch


_FACTORIES: dict[bool, SileroVadModelFactory] = {}
_FACTORIES_LOCK = threading.Lock()


def get_silero_vad_factory(*, use_onnx: bool) -> SileroVadModelFactory:
    with _FACTORIES_LOCK:
        factory = _FACTORIES.get(use_onnx)
        if factory is None:
            factory = SileroVadModelFactory(use_onnx=use_onnx)
            _FACTORIES[use_onnx] = factory
        return factory


def preload_silero_vad(*, use_onnx: bool = SILERO_VAD_ONNX) -> dict[str, Any]:
    get_silero_vad_factory(use_onnx=use_onnx)
    return silero_vad_status()


def silero_vad_status() -> dict[str, Any]:
    with _FACTORIES_LOCK:
        return {
            "preloaded": bool(_FACTORIES),
            "device": SILERO_DEVICE,
            "onnx_modes": [str(key).lower() for key in sorted(_FACTORIES.keys())],
        }


@dataclass(frozen=True)
class SileroVadConfig:
    threshold: float = SILERO_VAD_THRESHOLD
    min_speech_ms: int = SILERO_VAD_MIN_SPEECH_MS
    min_silence_ms: int = SILERO_VAD_MIN_SILENCE_MS
    max_speech_ms: int = SILERO_VAD_MAX_SPEECH_MS
    speech_pad_ms: int = SILERO_VAD_SPEECH_PAD_MS
    use_onnx: bool = SILERO_VAD_ONNX

    @property
    def neg_threshold(self) -> float:
        return max(self.threshold - 0.15, 0.01)


class SileroVadSession:
    def __init__(self, config: SileroVadConfig | None = None) -> None:
        self.config = config or SileroVadConfig()
        self._torch: Any
        self._model: Any
        self._load_model()
        self.reset()

    def _load_model(self) -> None:
        factory = get_silero_vad_factory(use_onnx=self.config.use_onnx)
        self._torch = factory.torch
        self._model = factory.new_model()

    def reset(self) -> None:
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()
        self.pending_samples: list[float] = []
        self.processed_samples = 0
        self.candidate_speech_samples = 0
        self.silence_samples = 0
        self.speech_started = False
        self.speech_start_sample = 0
        self.ended = False
        self.last_probability = 0.0

    def process_chunk(self, pcm_float32_bytes: bytes, source_sample_rate: int) -> list[dict[str, Any]]:
        if self.ended:
            return []

        samples = float32_bytes_to_list(pcm_float32_bytes)
        if source_sample_rate != SILERO_SAMPLE_RATE:
            samples = resample_linear(samples, source_sample_rate, SILERO_SAMPLE_RATE)
        self.pending_samples.extend(samples)

        events: list[dict[str, Any]] = []
        while len(self.pending_samples) >= SILERO_WINDOW_SAMPLES and not self.ended:
            frame = self.pending_samples[:SILERO_WINDOW_SAMPLES]
            del self.pending_samples[:SILERO_WINDOW_SAMPLES]
            events.extend(self._process_frame(frame))
        return events

    def _process_frame(self, frame: list[float]) -> list[dict[str, Any]]:
        frame_start = self.processed_samples
        frame_end = frame_start + SILERO_WINDOW_SAMPLES
        probability = self._speech_probability(frame)
        self.last_probability = probability
        self.processed_samples = frame_end

        if not self.speech_started:
            if probability >= self.config.threshold:
                self.candidate_speech_samples += SILERO_WINDOW_SAMPLES
                if self.candidate_speech_samples >= self._ms_to_samples(self.config.min_speech_ms):
                    self.speech_started = True
                    self.speech_start_sample = max(0, frame_end - self.candidate_speech_samples)
                    self.silence_samples = 0
                    return [
                        self._event(
                            "speech_start",
                            sample=self.speech_start_sample,
                            probability=probability,
                        )
                    ]
            elif probability < self.config.neg_threshold:
                self.candidate_speech_samples = 0
            return []

        if probability >= self.config.threshold:
            self.silence_samples = 0
        elif probability < self.config.neg_threshold:
            self.silence_samples += SILERO_WINDOW_SAMPLES

        speech_samples = frame_end - self.speech_start_sample
        if speech_samples >= self._ms_to_samples(self.config.max_speech_ms):
            return [self._finish("max_speech", frame_end, probability)]

        if self.silence_samples >= self._ms_to_samples(self.config.min_silence_ms):
            speech_end = max(
                self.speech_start_sample,
                frame_end - self.silence_samples + self._ms_to_samples(self.config.speech_pad_ms),
            )
            return [self._finish("speech_end", speech_end, probability)]

        return []

    def _speech_probability(self, frame: list[float]) -> float:
        tensor = self._torch.tensor(frame, dtype=self._torch.float32, device=SILERO_DEVICE)
        with self._torch.no_grad():
            return float(self._model(tensor, SILERO_SAMPLE_RATE).item())

    def _finish(self, reason: str, sample: int, probability: float) -> dict[str, Any]:
        self.ended = True
        return self._event(reason, sample=sample, probability=probability)

    def _event(self, event_type: str, sample: int, probability: float) -> dict[str, Any]:
        return {
            "event": event_type,
            "sample": int(sample),
            "time_ms": int(sample * 1000 / SILERO_SAMPLE_RATE),
            "probability": round(probability, 4),
            "threshold": self.config.threshold,
        }

    @staticmethod
    def _ms_to_samples(value_ms: int) -> int:
        return int(SILERO_SAMPLE_RATE * value_ms / 1000)
