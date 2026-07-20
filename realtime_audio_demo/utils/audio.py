import struct
import sys
import wave
from array import array
from io import BytesIO


def float32_sample_count(data: bytes) -> int:
    return len(data) // 4


def wav_bytes_from_float32_chunks(chunks: list[bytes], source_rate: int, target_rate: int) -> bytes:
    samples: list[float] = []
    for chunk in chunks:
        samples.extend(float32_bytes_to_list(chunk))
    if source_rate != target_rate:
        samples = resample_linear(samples, source_rate, target_rate)
    return pcm_float_to_wav_bytes(samples, target_rate)


def wav_bytes_list_to_wav_bytes(chunks: list[bytes], target_rate: int) -> bytes:
    samples: list[float] = []
    for wav_bytes in chunks:
        wav_samples, source_rate = wav_bytes_to_float_samples(wav_bytes)
        if source_rate != target_rate:
            wav_samples = resample_linear(wav_samples, source_rate, target_rate)
        samples.extend(wav_samples)
    return pcm_float_to_wav_bytes(samples, target_rate)


def float32_bytes_to_list(data: bytes) -> list[float]:
    arr = array("f")
    arr.frombytes(data[: len(data) - (len(data) % 4)])
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tolist()


def wav_bytes_to_float_samples(wav_bytes: bytes) -> tuple[list[float], int]:
    with wave.open(BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        frames = wf.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"unsupported wav sample width: {sample_width}")
    if channels != 1:
        raise ValueError(f"unsupported wav channels: {channels}")

    pcm = array("h")
    pcm.frombytes(frames[: len(frames) - (len(frames) % 2)])
    if sys.byteorder != "little":
        pcm.byteswap()
    samples = [max(-1.0, min(1.0, sample / 32768.0)) for sample in pcm]
    return samples, sample_rate


def resample_linear(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    if not samples or source_rate == target_rate:
        return samples
    output_length = max(1, int(len(samples) * target_rate / source_rate))
    ratio = source_rate / target_rate
    out: list[float] = []
    last_index = len(samples) - 1
    for i in range(output_length):
        pos = i * ratio
        left = int(pos)
        right = min(left + 1, last_index)
        frac = pos - left
        out.append(samples[left] * (1.0 - frac) + samples[right] * frac)
    return out


def pcm_float_to_wav_bytes(samples: list[float], sample_rate: int) -> bytes:
    frames = bytearray()
    for sample in samples:
        clipped = max(-1.0, min(1.0, sample))
        frames.extend(struct.pack("<h", int(clipped * 32767.0)))

    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(frames))
    return buf.getvalue()
